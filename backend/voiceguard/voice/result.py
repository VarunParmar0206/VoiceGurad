"""VoiceGuard V2 — Phase 5 typed audio / feature data structures.

These dataclasses define the stable contracts the pipeline produces and that
later phases (Phase 6 ML, client-side shared preprocessing) consume.  They
deliberately contain no raw PII and never serialize raw recordings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float32]


@dataclass
class DecodedAudio:
    """Audio decoded from an input source at its native format.

    This is the *input* to validation/canonicalization, not the canonical
    form.  ``samples`` is always 1-D or 2-D (samples, channels) floats in
    [-1.0, 1.0].
    """

    samples: FloatArray
    sample_rate: int
    channels: int

    @property
    def num_samples(self) -> int:
        return int(self.samples.shape[0])

    @property
    def duration_seconds(self) -> float:
        if self.sample_rate <= 0:
            return 0.0
        return self.num_samples / float(self.sample_rate)


@dataclass
class CanonicalWaveform:
    """Audio converted to the VoiceGuard canonical representation.

    Canonical contract:
      - ``samples`` is a 1-D float32 waveform
      - sample rate == target sample rate (configurable, default 16000)
      - mono (single channel)
      - amplitude in [-1.0, 1.0] with no NaN/Inf
    """

    samples: FloatArray
    sample_rate: int

    @property
    def num_samples(self) -> int:
        return int(self.samples.size)

    @property
    def duration_seconds(self) -> float:
        if self.sample_rate <= 0:
            return 0.0
        return self.num_samples / float(self.sample_rate)


@dataclass
class VoiceActivityResult:
    """Voice-activity detection + silence-trimming outcome."""

    voice_mask: npt.NDArray[np.bool_]
    start_sample: int
    end_sample: int  # exclusive
    voice_fraction: float
    was_trimmed: bool
    frames: int = 0
    frame_samples: int = 0


@dataclass
class PreprocessedAudio:
    """Framing + preprocessing outcome.

    ``frames`` is a 2-D array of shape ``(num_frames, frame_samples)`` in
    float32.  ``centered`` frames are typically windowed before FFT; the
    raw frame matrix is also kept for deterministic feature computation.
    """

    samples: FloatArray
    frames: FloatArray
    frame_samples: int
    hop_samples: int
    num_frames: int
    sample_rate: int
    vad: VoiceActivityResult


@dataclass
class MelFeatures:
    """Log-scaled mel-spectrogram and its tensor layout.

    Contract (see architecture §8.2 Path A):
      - ``log_mel`` shape ``(n_mels, T)`` where T ~ duration * sample_rate / hop
      - values are finite, log-scaled
      - deterministically computed for identical input
    """

    log_mel: FloatArray  # (n_mels, T)
    n_mels: int
    n_fft: int
    hop_length: int
    win_length: int
    sample_rate: int
    f_min: float
    f_max: float

    @property
    def time_frames(self) -> int:
        return int(self.log_mel.shape[1])


@dataclass
class StatisticalFeatures:
    """Per-utterance statistical/quality features (architecture §8.2 Path B).

    These are 1-D summary statistics (no temporal axis) used for GMM-style
    scoring and quality gating in later phases.
    """

    values: FloatArray  # 1-D float32, fixed known length per feature set
    names: list[str] = field(default_factory=list)

    @property
    def dimension(self) -> int:
        return int(self.values.size)


@dataclass
class QualityReport:
    """Quality / validation metadata describing how the input was handled."""

    input_duration_seconds: float
    canonical_duration_seconds: float
    peak_amplitude: float
    rms_amplitude: float
    voice_fraction: float
    trimmed: bool
    warnings: list[str] = field(default_factory=list)


@dataclass
class FeatureResult:
    """Structured output of the Phase 5 pipeline.

    This is the complete, self-describing result a Phase 6 consumer needs
    without knowing implementation details of preprocessing/feature
    extraction.  It never contains the raw waveform.
    """

    sample_rate: int
    num_samples: int
    duration_seconds: float
    mel: MelFeatures
    statistical: StatisticalFeatures
    quality: QualityReport
    meta: dict[str, Any] = field(default_factory=dict)
