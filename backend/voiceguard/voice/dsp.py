"""VoiceGuard V2 — Phase 5 internal DSP primitives.

Deterministic, dependency-light signal-processing helpers implemented with
numpy/scipy so feature extraction does not depend on librosa (whose numba
backend is incompatible with the pinned numpy release in this project).

All functions are pure, deterministic, and annotated with documented tensor
layouts.  This module is **internal** and not part of the public Phase 5 API.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy import signal as _signal

Float64Array = npt.NDArray[np.float64]

_HZ_TO_MEL = 2595.0
_MEL_TO_HZ = 700.0


def mel_from_hz(freq: npt.NDArray[np.float64] | float) -> npt.NDArray[np.float64]:
    """Convert frequency in Hz to the mel scale (HTK formula)."""
    return _HZ_TO_MEL * np.log10(1.0 + np.asarray(freq, dtype=np.float64) / 700.0)


def hz_from_mel(mel: npt.NDArray[np.float64] | float) -> npt.NDArray[np.float64]:
    """Convert the mel scale back to frequency in Hz."""
    return _MEL_TO_HZ * (10.0 ** (np.asarray(mel, dtype=np.float64) / _HZ_TO_MEL) - 1.0)


def mel_filterbank(
    n_mels: int,
    n_fft: int,
    sample_rate: int,
    f_min: float,
    f_max: float,
) -> npt.NDArray[np.float64]:
    """Triangular mel filterbank (HTK scale), shape ``(n_mels, n_fft // 2 + 1)``.

    Filters are normalized to unit area so they behave like a power-spectrum
    integration basis (matching standard DSP practice).
    """
    if n_mels < 1:
        raise ValueError("n_mels must be >= 1.")
    if not 0.0 <= f_min < f_max <= sample_rate / 2.0:
        raise ValueError(f"Invalid low/high band ({f_min}, {f_max}).")

    nbins = n_fft // 2 + 1
    freqs = np.linspace(0.0, float(sample_rate) / 2.0, nbins, dtype=np.float64)

    mel_low = mel_from_hz(f_min)
    mel_high = mel_from_hz(f_max)
    mel_points = np.linspace(mel_low, mel_high, n_mels + 2, dtype=np.float64)
    hz_points = hz_from_mel(mel_points)

    banks = np.zeros((n_mels, nbins), dtype=np.float64)
    for m in range(n_mels):
        left = hz_points[m]
        center = hz_points[m + 1]
        right = hz_points[m + 2]
        if center <= left or right <= center:
            continue
        up_idx = (freqs >= left) & (freqs <= center)
        down_idx = (freqs > center) & (freqs <= right)
        banks[m, up_idx] = (freqs[up_idx] - left) / (center - left)
        banks[m, down_idx] = (right - freqs[down_idx]) / (right - center)
    # Unit-area normalization per filter.
    norms = banks.sum(axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    banks /= norms
    return banks


def stft_magnitude_power(
    signal: npt.NDArray[np.float64],
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    window: str = "hann",
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Deterministic single-sided power spectrogram.

    Returns ``(freqs_hz, power)`` where ``power`` has shape
    ``(n_fft // 2 + 1, T)``.
    """
    n_fft, hop_length = int(n_fft), int(hop_length)
    if signal.size < 2:
        raise ValueError("signal must contain at least two samples.")
    # Deterministic zero-padding so very short signals still produce a full
    # n_fft analysis window (nperseg never shrinks below n_fft).
    if signal.size < n_fft:
        signal = np.concatenate([signal, np.zeros(n_fft - signal.size, dtype=np.float64)])
    freqs, _, zxx = _signal.stft(
        signal,
        fs=float(sample_rate),
        window=window,
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        boundary=None,
        padded=True,
    )
    power = np.abs(zxx) ** 2
    return freqs, power


def stft_power(
    signal: npt.NDArray[np.float64],
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    window: str = "hann",
) -> npt.NDArray[np.float64]:
    """Power spectrogram, shape ``(n_fft // 2 + 1, T)`` (frequency x time)."""
    _, power = stft_magnitude_power(signal, sample_rate, n_fft, hop_length, window)
    return power


def delta(features: npt.NDArray[np.float64], width: int = 9) -> npt.NDArray[np.float64]:
    """Local derivative of a (frames, dims) or (dims, frames) feature matrix.

    ``features`` layout ``(dims, F)`` (like librosa) is expected; edges are
    padded by repeating the boundary frame.  ``width`` must be odd.
    """
    feats = np.atleast_2d(features)
    if feats.ndim != 2:
        raise ValueError("delta expects a 2-D feature matrix.")
    width = int(width)
    if width < 3 or width % 2 != 1:
        raise ValueError(f"delta width must be odd and >= 3; got {width}.")
    half = (width - 1) // 2

    x = np.pad(feats, ((0, 0), (half, half)), mode="edge")
    denom = 2.0 * sum(i * i for i in range(1, half + 1))
    out = np.zeros_like(feats, dtype=np.float64)
    for d in range(1, half + 1):
        ahead = x[:, d + half : d + half + feats.shape[1]]
        behind = x[:, half - d : half - d + feats.shape[1]]
        out += d * (ahead - behind)
    return out / denom


def dct_type2(features: npt.NDArray[np.float64], n_coeffs: int) -> npt.NDArray[np.float64]:
    """Orthonormal type-II DCT along the first (feature) axis."""
    from scipy.fftpack import dct

    return np.asarray(dct(features, type=2, norm="ortho", axis=0)[:n_coeffs], dtype=np.float64)


def spectral_centroid(
    power: npt.NDArray[np.float64], freqs: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Per-frame spectral centroid when only `power` is present."""
    return _weighted_freq(power, freqs, 1) / _safe_sum(power)


def spectral_bandwidth(
    power: npt.NDArray[np.float64],
    freqs: npt.NDArray[np.float64],
    centroid: npt.NDArray[np.float64],
    p: float = 2,
) -> npt.NDArray[np.float64]:
    """Per-frame p-order spectral bandwidth."""
    diff = np.abs(freqs[:, None] - centroid[None, :]) ** p
    num = np.sum(power * diff, axis=0)
    den = _safe_sum(power)
    return (num / den) ** (1.0 / p)


def spectral_rolloff(
    power: npt.NDArray[np.float64],
    freqs: npt.NDArray[np.float64],
    roll_percent: float = 0.85,
) -> npt.NDArray[np.float64]:
    """Per-frame rolloff frequency at ``roll_percent`` cumulative power."""
    cumulative = np.cumsum(power, axis=0)
    total = cumulative[-1, :]
    total[total == 0.0] = 1.0
    frac = cumulative / total[None, :]
    idx = np.argmax(frac >= roll_percent, axis=0)
    return freqs[idx]


def spectral_flatness(power: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Per-frame spectral flatness (geometric/arithmetic mean of power)."""
    eps = 1e-10
    geom = np.exp(np.mean(np.log(power + eps), axis=0))
    arith = np.mean(power, axis=0)
    arith[arith == 0.0] = eps
    return geom / arith


def spectral_contrast(
    power: npt.NDArray[np.float64],
    freqs: npt.NDArray[np.float64],
    f_min: float,
    f_max: float,
    n_bands: int = 6,
) -> npt.NDArray[np.float64]:
    """Octave-band contrast: log gap between peak and valley band energies.

    Returns a single per-frame value (mean across bands used for the
    statistical vector is applied by the caller).
    """
    lo = max(1.0, f_min)
    hi = max(lo + 1.0, f_max)
    edges = np.logspace(np.log10(lo), np.log10(hi), n_bands + 1)
    values = np.zeros(power.shape[1], dtype=np.float64)
    for b in range(n_bands):
        band = np.flatnonzero((freqs >= edges[b]) & (freqs < edges[b + 1]))
        if band.size == 0:
            continue
        energy = power[band, :]
        if energy.size == 0:
            continue
        peak = np.mean(np.sort(energy, axis=0)[-int(max(1, band.size // 5)) :, :], axis=0)
        valley = np.mean(np.sort(energy, axis=0)[: int(max(1, band.size // 5)), :], axis=0)
        values += np.log(peak + 1e-9) - np.log(valley + 1e-9)
    return values / float(max(1, n_bands))


def zero_crossing_rate(
    signal: npt.NDArray[np.float64], frame: int, hop: int
) -> npt.NDArray[np.float64]:
    """Per-frame fraction of zero crossings."""
    n = int(signal.size)
    if n < 2:
        return np.zeros(max(1, n // max(1, hop) + 1), dtype=np.float64)
    num = max(1, (n - frame) // hop + 1)
    out = np.zeros(num, dtype=np.float64)
    for i in range(num):
        seg = signal[i * hop : i * hop + frame]
        if seg.size > 1:
            out[i] = np.mean(np.abs(np.diff(np.signbit(seg))))
    return out


def rms(signal: npt.NDArray[np.float64], frame: int, hop: int) -> npt.NDArray[np.float64]:
    """Per-frame root-mean-square energy."""
    n = int(signal.size)
    if n < 1:
        return np.zeros(1, dtype=np.float64)
    num = max(1, (n - frame) // hop + 1)
    out = np.zeros(num, dtype=np.float64)
    for i in range(num):
        seg = signal[i * hop : i * hop + frame]
        out[i] = np.sqrt(np.mean(np.square(seg))) if seg.size else 0.0
    return out


def autocorrelation_f0(
    signal: npt.NDArray[np.float64],
    sample_rate: int,
    frame: int = 320,  # 20 ms @ 16 kHz
    hop: int = 160,  # 10 ms @ 16 kHz
    f_min: float = 70.0,
    f_max: float = 400.0,
    voicing_threshold: float = 0.30,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    """Autocorrelation-based pitch tracking.

    Returns ``(f0_hz, voiced)`` with one value per frame.  Deterministic.
    """
    lo = int(round(float(sample_rate) / f_max))
    hi = int(round(float(sample_rate) / f_min))
    n = int(signal.size)
    num_frames = max(1, (n - frame) // hop + 1)
    f0 = np.zeros(num_frames, dtype=np.float64)
    voiced = np.zeros(num_frames, dtype=np.bool_)
    for i in range(num_frames):
        x = signal[i * hop : i * hop + frame]
        if x.size < lo:
            continue
        x = x - np.mean(x)
        var = float(np.dot(x, x))
        if var <= 1e-9:
            continue
        corr = np.correlate(x, x, mode="full")[x.size - 1 :]
        end = min(x.size, hi)
        window = corr[lo:end]
        if window.size == 0:
            continue
        peak_idx = int(np.argmax(window))
        peak_val = float(window[peak_idx])
        norm = peak_val / var
        if norm < voicing_threshold:
            continue
        f0[i] = float(sample_rate) / float(lo + peak_idx)
        voiced[i] = True
    return f0, voiced


def lpc(signal: npt.NDArray[np.float64], order: int) -> npt.NDArray[np.float64]:
    """Linear prediction coefficients (autocorrelation / Levinson-Durbin).

    Returns ``a`` of length ``order + 1`` with ``a[0] == 1.0``.
    """
    order = int(order)
    x = signal - np.mean(signal)
    n = int(x.size)
    maxlag = min(n - 1, order)
    ac = np.zeros(order + 1, dtype=np.float64)
    for k in range(order + 1):
        ac[k] = float(np.dot(x[: n - k], x[k:])) if n - k > 0 else 0.0
    r0 = ac[0]
    if r0 <= 1e-12:
        a = np.zeros(order + 1, dtype=np.float64)
        a[0] = 1.0
        return a
    a = np.zeros(order + 1, dtype=np.float64)
    a[0] = 1.0
    e = r0
    for m in range(1, maxlag + 1):
        acc = ac[m]
        for j in range(1, m):
            acc -= a[j] * ac[m - j]
        ref = acc / e
        a[m] = ref
        for j in range(1, m):
            a[j] = a[j] - ref * a[m - j]
        e *= 1.0 - ref * ref
    return a


def formants(
    signal: npt.NDArray[np.float64],
    sample_rate: int,
    order: int = 16,
    max_formants: int = 3,
) -> npt.NDArray[np.float64]:
    """Estimate formant frequencies per frame via LPC roots.

    Returns an array of shape ``(num_frames, max_formants)``; frames with
    fewer detected formants are zero-filled.  Output is in Hz.
    """
    frame = int(sample_rate * 0.020)
    hop = frame // 2
    n = int(signal.size)
    num_frames = max(1, (n - frame) // hop + 1)
    out = np.zeros((num_frames, max_formants), dtype=np.float64)
    for i in range(num_frames):
        x = signal[i * hop : i * hop + frame]
        a = lpc(x, min(order, max(2, x.size - 1)))
        roots = np.roots(a)
        freqs: list[float] = []
        for root in roots:
            if np.isreal(root):
                continue
            ang = np.abs(np.angle(root))
            if ang <= 0.0:
                continue
            freq = float(ang * sample_rate / (2.0 * np.pi))
            bandwidth = float(-np.log(np.abs(root)) * sample_rate / np.pi)
            if 90.0 < freq < sample_rate / 2.0 and bandwidth < 500.0:
                freqs.append(freq)
        freqs.sort()
        for j, freq in enumerate(freqs[:max_formants]):
            out[i, j] = freq
    return out


def _weighted_freq(
    power: npt.NDArray[np.float64], freqs: npt.NDArray[np.float64], exponent: float
) -> npt.NDArray[np.float64]:
    return np.sum(power * (freqs[:, None] ** exponent), axis=0)


def _safe_sum(power: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    total = np.sum(power, axis=0)
    total[total == 0.0] = 1e-12
    return total
