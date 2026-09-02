"""VoiceGuard V2 — Shared constants.

Values shared between backend and client that do not require
environment-variable overrides.
"""

from __future__ import annotations

# ── Audio Defaults ───────────────────────────────────────────────────────
DEFAULT_SAMPLE_RATE: int = 16000
DEFAULT_CHANNELS: int = 1

# ── Feature Extraction ──────────────────────────────────────────────────
DEFAULT_N_MFCC: int = 40
DEFAULT_N_MELS: int = 80
DEFAULT_HOP_LENGTH: int = 160
DEFAULT_N_FFT: int = 512
DEFAULT_F_MIN: int = 50
DEFAULT_F_MAX: int = 8000

# ── Enrollment ───────────────────────────────────────────────────────────
ENROLLMENT_PHRASES: list[str] = [
    "Authorize payment",
    "Verify transaction",
    "Complete purchase",
    "Confirm identity",
    "Voice authentication",
]

# ── Challenge Vocabulary (subset — full list lives in backend) ───────────
CHALLENGE_PHRASES: list[str] = [
    "Verify the current session",
    "Confirm this transaction now",
    "Authorize the payment amount",
    "Validate voice identity check",
    "Process the secure transfer",
    "Authenticate the requested action",
    "Approve the pending payment",
    "Confirm the transfer details",
    "Verify account ownership",
    "Authorize the secure transaction",
]

# ── Currency ─────────────────────────────────────────────────────────────
CURRENCY_SYMBOL: str = "₹"
CURRENCY_CODE: str = "INR"
