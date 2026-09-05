"""VoiceGuard V2 — Phase 5 → Phase 6 ML input adapter.

Converts a :class:`voiceguard.voice.result.FeatureResult` (log-mel
``(80, T)`` float32, canonical 16 kHz) into the fixed-layout tensor the
:class:`CNNLSTMAttention` model consumes: ``(batch, 1, n_mels=80, T)``
with explicit per-utterance validity so padding never pollutes attention.

Layout / normalization policy (documented, identical for enrollment and
verification — the same function is used on both sides):

- ``T > max_frames`` → truncated to the **first** ``max_frames`` frames
  (column 0 is the start of the utterance; 300 frames ≈ 3 s at 16 kHz/160).
- ``T < max_frames`` → columns ``T..max_frames-1`` are set to zero.
- Per-band mean/std normalization (when enabled) is computed over **valid**
  frames only, so zero-padding cannot shift the statistics.  Padded columns
  are pinned to ``0.0`` after normalization.
- The ``lengths`` tensor lets the model pack sequences and mask attention so
  padded frames receive zero weight.

The adapter never touches raw audio — it only consumes the already-computed
``FeatureResult`` structures.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
import numpy.typing as npt
import torch

from voiceguard.config import settings
from voiceguard.ml.errors import EmbeddingContractError
from voiceguard.voice.result import FeatureResult

FloatArray = npt.NDArray[np.float32]
Tensor = torch.Tensor

_NORM_EPS = 1e-6


@dataclass(frozen=True)
class ModelInput:
    """Fixed-layout batched model input produced by the adapter.

    Attributes
    ----------
    mel : Tensor
        ``(batch, 1, n_mels, time_frames)`` float32 log-mel (padded/capped).
    lengths : Tensor
        ``(batch,)`` int64 valid frame counts measured on the **input**
        resolution (pre-CNN pooling), for packing + attention masking.
    time_frames : int
        Padded/truncated time axis length after the adapter.
    n_mels : int
        Frequency dimension (80).
    truncated : bool
        At least one utterance exceeded ``max_frames``.
    padded : bool
        At least one utterance was shorter than ``time_frames``.
    normalized : bool
        Whether per-band normalization was applied over valid frames.
    max_frames : int
        The configured frame cap used to build this input.
    """

    mel: Tensor
    lengths: Tensor
    time_frames: int
    n_mels: int
    truncated: bool
    padded: bool
    normalized: bool
    max_frames: int


def prepare_input(
    result: FeatureResult,
    *,
    max_frames: int | None = None,
    normalize_mel: bool | None = None,
    device: torch.device | None = None,
) -> ModelInput:
    """Adapt a single ``FeatureResult`` to the model input contract.

    Parameters
    ----------
    result : FeatureResult
        Phase 5 output (must contain a valid canonical mel spectrogram).
    max_frames : int | None
        Column cap; defaults to ``settings.MAX_EMBEDDING_FRAMES`` (300).
    normalize_mel : bool | None
        Override for per-band normalization; defaults to
        ``settings.EMBEDDING_MEL_NORMALIZE``.  Must be consistent across
        enrollment and verification.
    device : torch.device | None
        Device for the returned tensors (default: CPU).

    Raises
    ------
    EmbeddingContractError
        If the ``FeatureResult`` violates the Phase 5 → 6 mel contract.
    """
    cap = settings.MAX_EMBEDDING_FRAMES if max_frames is None else max_frames
    if cap < 1:
        raise EmbeddingContractError("max_frames must be >= 1")

    normalize = (
        settings.EMBEDDING_MEL_NORMALIZE
        if normalize_mel is None
        else bool(normalize_mel)
    )

    _validate_contract(result)

    log_mel = result.mel.log_mel  # (n_mels, T) float32
    n_mels = int(log_mel.shape[0])
    t_valid = int(log_mel.shape[1])

    truncated = t_valid > cap
    padded = t_valid < cap
    time_frames = cap

    # Work on a copy so the caller's FeatureResult is never mutated.
    # The layout is ALWAYS (1, 1, n_mels, cap): valid frames occupy columns
    # [0, min(t_valid, cap)), the rest are pinned to zero and excluded from
    # the length mask so the model ignores them.
    band = np.zeros((n_mels, cap), dtype=np.float32)
    band[:, : min(t_valid, cap)] = np.asarray(
        log_mel[:, :cap], dtype=np.float32
    )[:, : min(t_valid, cap)]

    if normalize and t_valid > 0:
        valid = band[:, : min(t_valid, cap)]
        mean = valid.mean(axis=1, keepdims=True)
        std = valid.std(axis=1, keepdims=True)
        denom = np.maximum(std, _NORM_EPS)
        band = ((band - mean) / denom).astype(np.float32)
        band[:, min(t_valid, cap) :] = 0.0

    # Assemble (1, 1, n_mels, time_frames) float32 contiguous tensor.
    sample = np.ascontiguousarray(band[np.newaxis, np.newaxis, :, :])
    mel = torch.from_numpy(sample)
    length = torch.tensor([min(t_valid, cap)], dtype=torch.long)
    if device is not None:
        mel = mel.to(device)
        length = length.to(device)

    return ModelInput(
        mel=mel,
        lengths=length,
        time_frames=time_frames,
        n_mels=n_mels,
        truncated=truncated,
        padded=padded,
        normalized=normalize,
        max_frames=cap,
    )


def batch_inputs(
    inputs: Sequence[ModelInput],
    *,
    device: torch.device | None = None,
) -> ModelInput:
    """Stack pre-adapted inputs into a single batched ``ModelInput``.

    All inputs must share the same ``n_mels`` and ``time_frames`` (they are
    produced by ``prepare_input`` with a consistent ``max_frames``).
    """
    if not inputs:
        raise EmbeddingContractError("cannot batch an empty input sequence")
    first = inputs[0]
    if any(i.n_mels != first.n_mels for i in inputs):
        raise EmbeddingContractError("all inputs must share the same n_mels")
    if any(i.time_frames != first.time_frames for i in inputs):
        raise EmbeddingContractError(
            "all inputs must share the same time_frames (use a consistent max_frames)"
        )

    mel = torch.cat([i.mel for i in inputs], dim=0)
    lengths = torch.cat([i.lengths for i in inputs], dim=0)
    if device is not None:
        mel = mel.to(device)
        lengths = lengths.to(device)

    return ModelInput(
        mel=mel,
        lengths=lengths,
        time_frames=first.time_frames,
        n_mels=first.n_mels,
        truncated=any(i.truncated for i in inputs),
        padded=any(i.padded for i in inputs),
        normalized=first.normalized,
        max_frames=first.max_frames,
    )


def embed_result(
    result: FeatureResult,
    model: torch.nn.Module,
    *,
    max_frames: int | None = None,
    normalize_mel: bool | None = None,
) -> FloatArray:
    """Encode one ``FeatureResult`` into a speaker embedding (eager, CPU).

    Returns a float32 1-D array of size ``model.embedding_dim`` (256).  The
    model is switched to evaluation mode (dropout/BatchNorm disabled) and the
    forward pass runs under ``torch.inference_mode`` so results are
    deterministic for a fixed model + input.

    Raises
    ------
    EmbeddingContractError
        On any contract violation.
    voiceguard.ml.errors.ModelError
        On any model-level input failure.
    """
    prepared = prepare_input(result, max_frames=max_frames, normalize_mel=normalize_mel)
    embedding = embed_model_input(prepared, model, batch_index=0)
    return embedding


def embed_batch(
    results: Sequence[FeatureResult],
    model: torch.nn.Module,
    *,
    max_frames: int | None = None,
    normalize_mel: bool | None = None,
) -> FloatArray:
    """Encode many ``FeatureResult`` objects into ``(batch, d)`` float32."""
    if not results:
        raise EmbeddingContractError("cannot embed an empty sequence")
    prepared = batch_inputs(
        [
            prepare_input(r, max_frames=max_frames, normalize_mel=normalize_mel)
            for r in results
        ]
    )
    return embed_model_input(prepared, model)


def embed_model_input(
    prepared: ModelInput, model: torch.nn.Module, *, batch_index: int | None = None
) -> FloatArray:
    """Run the model on a prepared input and return numpy float32 embedding(s).

    If ``batch_index`` is given, a single 1-D embedding is returned; otherwise
    a ``(batch, d)`` matrix.
    """
    device = _module_device(model)
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            out = model(prepared.mel.to(device), prepared.lengths.to(device))
    finally:
        if was_training:
            model.train()
    if batch_index is not None:
        return cast(FloatArray, out[batch_index].detach().cpu().numpy().astype(np.float32))
    return cast(FloatArray, out.detach().cpu().numpy().astype(np.float32))


def _module_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _validate_contract(result: FeatureResult) -> None:
    mel = result.mel
    if result.sample_rate != settings.AUDIO_SAMPLE_RATE:
        raise EmbeddingContractError(
            f"FeatureResult sample_rate is {result.sample_rate}, expected "
            f"{settings.AUDIO_SAMPLE_RATE}"
        )
    if mel.n_mels != settings.N_MELS:
        raise EmbeddingContractError(
            f"mel has {mel.n_mels} bands, expected {settings.N_MELS}"
        )
    if mel.hop_length != settings.HOP_LENGTH:
        raise EmbeddingContractError(
            f"mel hop_length is {mel.hop_length}, expected {settings.HOP_LENGTH}"
        )
    if mel.n_fft != settings.N_FFT:
        raise EmbeddingContractError(
            f"mel n_fft is {mel.n_fft}, expected {settings.N_FFT}"
        )
    if mel.log_mel.ndim != 2:
        raise EmbeddingContractError(
            f"mel must be 2-D (n_mels, T); got {mel.log_mel.ndim}D"
        )
    if mel.log_mel.shape[0] != settings.N_MELS:
        raise EmbeddingContractError(f"mel frequency dim is {mel.log_mel.shape[0]}")
    if mel.log_mel.dtype != np.float32:
        raise EmbeddingContractError(
            f"mel dtype must be float32; got {mel.log_mel.dtype}"
        )
    if mel.time_frames < 1:
        raise EmbeddingContractError("mel has no time frames")
    if not np.isfinite(mel.log_mel).all():
        raise EmbeddingContractError("mel contains non-finite values")
