"""VoiceGuard V2 — CNN-LSTM-Attention speaker embedding model.

Implements architecture §6.2.

Input : ``(batch, 1, n_mels, T)`` log-mel spectrograms (float32), where
        ``T`` is a fixed padded/capped length (the adapter caps it to at most
        ``MAX_EMBEDDING_FRAMES`` frames).  Variable-length info is carried in
        an optional ``lengths`` tensor (valid frames per utterance, BEFORE the
        CNN time-pooling), which drives sequence packing so padded frames do
        not corrupt the bidirectional LSTM, plus an attention time mask so
        padded frames receive zero weight.

Output : ``(batch, embedding_dim=256)``, L2-normalized.

The model is trained with triplet + ArcFace losses (see
``voiceguard.ml.training``); a full forward pass runs in eager mode on CPU so
the architecture is exercisable without a GPU.

Note
****
This replaces the obsolete V1 ``VoiceEmbeddingNet`` (random weights, fixed
256-dim vector input).  No legacy model is part of the public Phase 6 API.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from voiceguard.ml.errors import ModelError

Tensor = torch.Tensor


def pool_length(length: int, kernel: int = 2, stride: int = 2) -> int:
    """MaxPool2d output size for a 1-D dimension (matches PyTorch rounding).

    PyTorch computes ``floor((length - kernel) / stride) + 1`` and clamps to
    a minimum of 1 so that very small inputs do not collapse to zero.
    """
    if length <= 0:
        raise ValueError("length must be a positive integer")
    return max(1, (length - kernel) // stride + 1)


def _pooled_sequence_lengths(lengths: Tensor) -> Tensor:
    """Map valid pre-CNN time lengths to post-pooling attention lengths."""
    return torch.as_tensor(
        [pool_length(pool_length(int(length))) for length in lengths.detach().tolist()],
        device=lengths.device,
        dtype=torch.long,
    )


class CNNLSTMAttention(nn.Module):
    """CNN front-end → bidirectional LSTM → additive attention → embedding.

    Parameters
    ----------
    n_mels : int
        Frequency axis of the input log-mel spectrogram (default 80).
    embedding_dim : int
        Output embedding dimensionality (default 256).
    cnn_channels : tuple[int, int, int]
        Channel count per CNN block (default 32 → 64 → 128).
    cnn_dropout : float
        Dropout applied after the final CNN block (architecture §6.3: 0.2).
    lstm_hidden : int
        Hidden units per direction (default 256 → 512 concatenated).
    lstm_layers : int
        Number of stacked BiLSTM layers (default 2).
    lstm_dropout : float
        Dropout between LSTM layers (architecture §6.3: 0.3).
    """

    def __init__(
        self,
        n_mels: int = 80,
        embedding_dim: int = 256,
        cnn_channels: tuple[int, int, int] = (32, 64, 128),
        cnn_dropout: float = 0.2,
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        lstm_dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if len(cnn_channels) != 3:
            raise ValueError("cnn_channels must have exactly three entries")

        c1, c2, c3 = cnn_channels
        self.n_mels = int(n_mels)
        self.embedding_dim = int(embedding_dim)
        self.lstm_hidden = int(lstm_hidden)
        projection_dim = lstm_hidden * 2  # bidirectional concat (512)

        # ── CNN front-end ───────────────────────────────────────────────
        self.conv1 = self._conv_block(1, c1)
        self.conv2 = self._conv_block(c1, c2)
        self.conv3 = self._conv_block(c2, c3, pool=False, dropout=cnn_dropout)
        # Preserve the temporal axis, collapse frequency to a single channel.
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, None))

        # ── Bi-LSTM sequence modeling ────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=c3,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout=lstm_dropout,
            batch_first=True,
            bidirectional=True,
        )

        # ── Projection + additive (Bahdanau) attention ──────────────────
        self.projection = nn.Linear(projection_dim, projection_dim)
        self.attn_w = nn.Linear(projection_dim, projection_dim, bias=False)
        self.attn_v = nn.Parameter(torch.empty(projection_dim))
        # Final embedding projection.
        self.embed = nn.Linear(projection_dim, embedding_dim)

        self._reset_attention_parameters()

    @staticmethod
    def _conv_block(
        in_channels: int,
        out_channels: int,
        pool: bool = True,
        dropout: float = 0.0,
    ) -> nn.Sequential:
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout2d(dropout))
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2))
        return nn.Sequential(*layers)

    def _reset_attention_parameters(self) -> None:
        # Bound initialization for the additive attention energy vector.
        nn.init.uniform_(self.attn_v, -0.01, 0.01)

    # ── Public API ──────────────────────────────────────────────────────

    def forward(self, mel: Tensor, lengths: Tensor | None = None) -> Tensor:
        """Embed a batch of log-mel spectrograms.

        Parameters
        ----------
        mel : Tensor
            ``(batch, 1, n_mels, T)`` float32 log-mel spectrogram.
        lengths : Tensor | None
            ``(batch,)`` int64 valid (unpadded) time-frame counts measured on
            the **input** resolution (before CNN time-pooling).  When given,
            sequence packing + a masked additive attention prevent padded
            frames from contributing to the embedding.

        Returns
        -------
        Tensor
            ``(batch, embedding_dim)`` L2-normalized embeddings.
        """
        batch = mel.shape[0]
        if mel.dim() != 4:
            raise ModelError(f"mel must be 4-D (B, 1, n_mels, T); got {mel.dim()}D")
        if mel.shape[1] != 1:
            raise ModelError(f"mel channel dim must be 1; got {mel.shape[1]}")
        if mel.shape[2] != self.n_mels:
            raise ModelError(
                f"mel frequency dim must be {self.n_mels}; got {mel.shape[2]}"
            )
        t_in = mel.shape[3]
        if t_in < 4:
            raise ModelError(
                f"mel time axis must be >= 4 frames for valid 2x pooling; got {t_in}"
            )
        if not torch.isfinite(mel).all():
            raise ModelError("mel contains non-finite values")
        if lengths is not None:
            if lengths.dim() != 1 or lengths.shape[0] != batch:
                raise ModelError(
                    f"lengths must be (batch,) with batch={batch}; got {tuple(lengths.shape)}"
                )
            if torch.any(lengths < 1) or torch.any(lengths > t_in):
                raise ModelError("lengths must satisfy 1 <= lengths <= T")

        # CNN front-end: (B, 1, 80, T) -> (B, C3, 1, T') after adaptive pool
        # over frequency.
        x = self.conv3(self.conv2(self.conv1(mel)))
        x = self.adaptive_pool(x)  # (B, C3, 1, T')
        x = x.squeeze(2)           # (B, C3, T')
        x = x.transpose(1, 2).contiguous()  # (B, T', C3)

        t_pooled = x.shape[1]

        # BiLSTM over valid frames only (packed) so trailing padding cannot
        # corrupt the backward pass.
        if lengths is not None:
            packed = pack_padded_sequence(
                x,
                _pooled_sequence_lengths(lengths).cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            x, _ = pad_packed_sequence(packed, batch_first=True, total_length=t_pooled)
            x = x.contiguous()

        lstm_out, _ = self.lstm(x)  # (B, T', 512)
        hidden = F.relu(self.projection(lstm_out))  # (B, T', 256)

        # Additive attention: alpha = softmax_t( tanh(W_a h) . v )
        energy = torch.tanh(self.attn_w(hidden)) @ self.attn_v  # (B, T')
        if lengths is not None:
            pooled_len = _pooled_sequence_lengths(lengths)  # (B,)
            mask = torch.arange(t_pooled, device=energy.device)[None, :] < pooled_len[:, None]
            energy = energy.masked_fill(~mask, float("-inf"))
        alpha = torch.softmax(energy, dim=1)
        context = (alpha.unsqueeze(-1) * hidden).sum(dim=1)  # (B, 256)

        embedding = F.normalize(self.embed(context), p=2, dim=1)
        return embedding

    def attention_weights(self, mel: Tensor, lengths: Tensor | None = None) -> Tensor:
        """Return the additive-attention weights ``(batch, T_pooled)``.

        Each row is a distribution over time frames (sums to 1); padded frames
        carry zero weight.  Useful for debugging and for tests asserting the
        time-mask behaviour.
        """
        if mel.dim() != 4 or mel.shape[1] != 1 or mel.shape[2] != self.n_mels:
            raise ModelError("mel must be (B, 1, n_mels, T)")
        x = self.conv3(self.conv2(self.conv1(mel)))
        x = self.adaptive_pool(x)
        x = x.squeeze(2).transpose(1, 2).contiguous()
        t_pooled = x.shape[1]
        if lengths is not None:
            packed = pack_padded_sequence(
                x,
                _pooled_sequence_lengths(lengths).cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            x, _ = pad_packed_sequence(packed, batch_first=True, total_length=t_pooled)
            x = x.contiguous()
        hidden = F.relu(self.projection(self.lstm(x)[0]))
        energy = torch.tanh(self.attn_w(hidden)) @ self.attn_v
        if lengths is not None:
            pooled_len = _pooled_sequence_lengths(lengths)
            mask = torch.arange(t_pooled, device=energy.device)[None, :] < pooled_len[:, None]
            energy = energy.masked_fill(~mask, float("-inf"))
        return torch.softmax(energy, dim=1)
