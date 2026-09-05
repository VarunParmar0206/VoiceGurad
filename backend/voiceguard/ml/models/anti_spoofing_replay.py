"""VoiceGuard V2 — Phase 7 replay-attack detector (architecture §10).

A lightweight 1-D CNN (3 convolutional blocks) followed by a fully-connected
binary classifier that decides **live** vs. **replay** from a mel-spectrogram
frame sequence.

Design constraints
******************
- Input: ``mel`` of shape ``(B, 1, n_mels, T)`` — the same tensor layout
  produced by :func:`voiceguard.voice.embedding.prepare_input`.
- Output: ``(B, 1)`` logit and ``(B, 1)`` probability in ``[0, 1]``.
- Target inference: **~10 ms on CPU** for a typical 3-second utterance.
- No raw audio is ever consumed or stored by this module.

Training data
*************
Synthetic test data only (Phase 7 scaffold).  Real replay detection
training requires ASVspoof 2019/2021 datasets; the training pipeline
(:mod:`voiceguard.ml.training.train_replay`) is compatible with that
data once it becomes available.

Disclaimers
***********
No production AUC-ROC, ACER, BPCER, or APCER is claimed.  All metrics
produced from synthetic data are **test-only artifacts**.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from voiceguard.ml.errors import AntiSpoofError

Tensor = torch.Tensor


def pool_length(length: int) -> int:
    """Compute the output time-axis length after a ``MaxPool1d(2, 2)``."""
    return max(0, (length - 2) // 2 + 1)


class ReplayDetector(nn.Module):
    """1-D CNN binary replay detector (architecture §10.2).

    The network processes a mel-spectrogram through three 1-D convolutional
    blocks (each: Conv1d → BatchNorm → ReLU → MaxPool1d), applies global
    average pooling, and feeds the result through a two-layer classifier.

    Parameters
    ----------
    n_mels : int
        Number of mel frequency bins (default 80).
    cnn_channels : tuple[int, int, int]
        Channel counts for the three convolutional blocks.
    dropout : float
        Dropout applied between classifier layers.
    """

    def __init__(
        self,
        n_mels: int = 80,
        cnn_channels: tuple[int, int, int] = (32, 64, 128),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if len(cnn_channels) != 3:
            raise AntiSpoofError("cnn_channels must have exactly three entries")
        self.n_mels = int(n_mels)
        c1, c2, c3 = cnn_channels

        # ── 1-D CNN front-end ────────────────────────────────────────────
        # Input: (B, n_mels, T) — each mel bin is a channel, conv1d
        # processes over the time axis.
        self.conv1 = self._conv_block(n_mels, c1)
        self.conv2 = self._conv_block(c1, c2)
        self.conv3 = self._conv_block(c2, c3, pool=False)

        # ── Classifier ───────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(c3, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    @staticmethod
    def _conv_block(
        in_channels: int,
        out_channels: int,
        pool: bool = True,
    ) -> nn.Sequential:
        layers: list[nn.Module] = [
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool1d(kernel_size=2))
        return nn.Sequential(*layers)

    def forward(self, mel: Tensor) -> tuple[Tensor, Tensor]:
        """Classify a mel-spectrogram batch as live or replay.

        Parameters
        ----------
        mel : Tensor
            Shape ``(B, 1, n_mels, T)`` — the standard adapter output.

        Returns
        -------
        logits : Tensor
            Shape ``(B, 1)`` raw logit scores.
        probs : Tensor
            Shape ``(B, 1)`` sigmoid probabilities in ``[0, 1]``.
        """
        if mel.dim() != 4 or mel.shape[1] != 1 or mel.shape[2] != self.n_mels:
            raise AntiSpoofError(
                f"mel must be (B, 1, {self.n_mels}, T); got {tuple(mel.shape)}"
            )

        x = mel.squeeze(1)  # (B, n_mels, T)

        if x.shape[2] < 4:
            # Very short time axis: fall back to mean pooling over time.
            x = x.mean(dim=2)  # (B, n_mels)
            # Project to c3 channels.
            c3 = self.classifier[0].in_features
            x = x[:, :c3] if x.shape[1] >= c3 else torch.nn.functional.pad(
                x, (0, c3 - x.shape[1])
            )
        else:
            x = self.conv1(x)  # (B, 32, T')
            x = self.conv2(x)  # (B, 64, T'')
            x = self.conv3(x)  # (B, 128, T''')
            # Global average pooling over time.
            x = x.mean(dim=2)  # (B, 128)

        logits = self.classifier(x)  # (B, 1)
        probs = torch.sigmoid(logits)
        return logits, probs

    def predict(
        self, mel: Tensor, *, threshold: float = 0.5
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Classify and return decisions.

        Parameters
        ----------
        mel : Tensor
            Shape ``(B, 1, n_mels, T)``.
        threshold : float
            Probability threshold above which a sample is classified as
            **replay**.

        Returns
        -------
        logits : Tensor
            Shape ``(B, 1)``.
        probs : Tensor
            Shape ``(B, 1)``.
        is_replay : Tensor
            Shape ``(B, 1)`` boolean — ``True`` where prob >= threshold.
        """
        logits, probs = self.forward(mel)
        is_replay = probs >= threshold
        return logits, probs, is_replay
