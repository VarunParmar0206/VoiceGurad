"""VoiceGuard V2 — Centralized configuration.

All application constants are defined here as Pydantic BaseSettings fields.
Values are loaded from environment variables with the ``VG_`` prefix,
falling back to defaults where appropriate.

Required secrets (no defaults) must be set via environment variables or a
``.env`` file:
  - ``VG_DATABASE_URL``
  - ``VG_JWT_SECRET_KEY``
  - ``VG_ENCRYPTION_KEY``
"""

from __future__ import annotations

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # ── Application ──────────────────────────────────────────────────────
    APP_NAME: str = "VoiceGuard"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    # ── Server ───────────────────────────────────────────────────────────
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1

    # ── Database ─────────────────────────────────────────────────────────
    DATABASE_URL: str = Field(
        ...,
        description="Async PostgreSQL connection string, e.g. "
        "postgresql+asyncpg://user:pass@localhost:5432/voiceguard",
    )
    REDIS_URL: str = "redis://localhost:6379/0"

    # ── Security ─────────────────────────────────────────────────────────
    JWT_SECRET_KEY: str = Field(
        ...,
        description="Secret key used to sign JWT access/refresh tokens.",
    )
    JWT_ALGORITHM: str = "HS256"
    JWT_ISSUER: str = "voiceguard"
    JWT_AUDIENCE: str = "voiceguard"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 15
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    ENCRYPTION_KEY: str = Field(
        ...,
        description="Base64-encoded Fernet key for data-at-rest encryption.",
    )

    # ── TOTP (backup / secondary MFA) ─────────────────────────────────────
    TOTP_ISSUER: str = "VoiceGuard"
    TOTP_CODE_DIGITS: int = 6
    TOTP_SECRET_BYTES: int = 20  # 160-bit random key (20 bytes)
    TOTP_WINDOW_STEPS: int = 1  # ±1 time-step tolerance for clock skew

    # ── Session management ────────────────────────────────────────────────
    CONCURRENT_SESSION_LIMIT: int = 3

    # ── Two-step login ────────────────────────────────────────────────────
    # Lifetime (seconds) of the server-side one-time login state issued after
    # the password step and consumed by the secondary factor (TOTP).
    PENDING_LOGIN_EXPIRE_SECONDS: int = 300

    # ── Audio / Feature Extraction ───────────────────────────────────────
    AUDIO_SAMPLE_RATE: int = 16000
    AUDIO_CHANNELS: int = 1
    N_MFCC: int = 40
    N_MELS: int = 80
    HOP_LENGTH: int = 160
    N_FFT: int = 512
    WIN_LENGTH: int = 512  # 0/absent not possible in pydantic; equals N_FFT default
    F_MIN: int = 50
    F_MAX: int = 8000
    MEL_FLOOR: float = 1e-6  # additive floor in log(mel + floor)

    # ── ML / Speaker Verification ────────────────────────────────────────
    # Deterministic seed shared by model initialization and training.
    SEED: int = 42
    # Learner embedding dimensionality (architecture §6.2).
    EMBEDDING_DIM: int = 256
    ENROLLMENT_MIN_SAMPLES: int = 5
    # Single base verification threshold (architecture §5.2).
    #
    # UNCALIBRATED DESIGN DEFAULT: this value is inherited from the design
    # documents and has not been validated against real speaker data.
    # Treat it as a placeholder until a real dataset permits EER/threshold
    # calibration — it MUST NOT be cited as a production decision boundary.
    VERIFICATION_THRESHOLD: float = 0.82
    # Scores in [threshold - CONFIDENCE_BAND, threshold) trigger a soft
    # accept (request an additional sample) rather than a hard reject.
    CONFIDENCE_BAND: float = 0.05
    # Composite score weights (cosine / mahalanobis / gmm). Uncalibrated
    # design defaults; distinct from graduated-threshold logic (removed).
    COMPOSITE_WEIGHT_COSINE: float = 0.6
    COMPOSITE_WEIGHT_MAHALANOBIS: float = 0.3
    COMPOSITE_WEIGHT_GMM: float = 0.1
    # Per-user adaptive-threshold behaviour (architecture §5.2): the margin
    # added to the base threshold scales with (1 - intra-class consistency).
    ENROLLMENT_ADAPTIVE_FACTOR: float = 0.05
    ENROLLMENT_THRESHOLD_LOWER: float = 0.5
    ENROLLMENT_THRESHOLD_UPPER: float = 0.95
    # Per-user diagonal GMM (architecture §5.2/§24).  Fitted component count
    # is clamped to the number of enrollment samples at fit time; synthetic
    # models are scaffolding only — never claimed as validated.
    GMM_N_COMPONENTS: int = 8
    GMM_COVARIANCE_TYPE: str = "diag"
    GMM_SCALE_COMPONENTS: bool = True  # clamp components to n_samples
    # Statistical feature vector (Path B) dimensionality: 259 = 240 MFCC +
    # 5 spectral + 2 zcr + 4 pitch + 2 rms + 6 formants.
    STATISTICAL_FEATURE_DIM: int = 259
    # Phase 6 adapter policy (architecture §6.2): mel T is capped/padded to
    # at most ~300 frames (~3 s at 16 kHz / 160-hop).  Padding is zero-valued
    # and excluded from attention via a time mask; statistics are computed on
    # valid frames only.
    MAX_EMBEDDING_FRAMES: int = 300
    # Whether the adapter applies per-band mean/std normalization to mel
    # columns.  The SAME policy must apply to enrollment and verification.
    EMBEDDING_MEL_NORMALIZE: bool = True
    # Cancelable-transform PBKDF2-HMAC-SHA256 iteration count.  A derivable
    # design default; deploy-time tuning is expected.
    CANCELABLE_PBKDF2_ITERATIONS: int = 100_000
    CANCELABLE_KEY_BYTES: int = 32
    CANCELABLE_SALT_BYTES: int = 16
    SESSION_TIMEOUT_SECONDS: int = 900  # 15 minutes

    # ── Audio Quality ────────────────────────────────────────────────────
    MIN_AUDIO_DURATION_SECONDS: float = 0.5
    MAX_AUDIO_DURATION_SECONDS: float = 30.0
    MAX_AUDIO_SAMPLES: int = 0  # 0 = computed from MAX_AUDIO_DURATION_SECONDS * sample_rate
    MIN_QUALITY_SCORE: float = 0.45
    VAD_ENERGY_THRESHOLD: float = 0.02
    PRE_EMPHASIS_COEFFICIENT: float = 0.97
    PRE_EMPHASIS_ENABLED: bool = False  # Only enable for formant/LPC analysis paths
    PEAK_TARGET_DBFS: float = -1.0
    VAD_MIN_SPEECH_SECONDS: float = 0.5
    VAD_SILENCE_MARGIN_SECONDS: float = 0.05
    VAD_FILL_GAP_SECONDS: float = 0.15
    SILENCE_FLOOR_DBFS: float = -40.0

    # ── Anti-Spoofing ────────────────────────────────────────────────────
    ANTI_SPOOF_THRESHOLD: float = 0.70
    REPLAY_DETECTION_THRESHOLD: float = 0.90
    CHALLENGE_EXPIRY_SECONDS: int = 30
    CHALLENGE_MAX_PER_MINUTE: int = 3
    WER_PASS_THRESHOLD: float = 0.2
    WER_MARGINAL_THRESHOLD: float = 0.4

    # ── Rate Limiting ────────────────────────────────────────────────────
    RATE_LIMIT_AUTH_PER_MINUTE: int = 5
    RATE_LIMIT_VOICE_PER_MINUTE: int = 10
    RATE_LIMIT_TRANSACTION_PER_HOUR: int = 10

    # ── Account ──────────────────────────────────────────────────────────
    MAX_FAILED_ATTEMPTS: int = 5
    LOCKOUT_COOLDOWN_BASE_SECONDS: int = 30
    LOCKOUT_COOLDOWN_MAX_SECONDS: int = 300
    DAILY_TRANSACTION_LIMIT: float = 50000.00
    DEFAULT_BALANCE: float = 10000.00

    # ── Model Registry ──────────────────────────────────────────────────
    MODEL_STORAGE_PATH: str = "models/"
    # Current model/artifact version tag written into registry metadata.
    MODEL_VERSION: str = "v1.0"

    @field_validator("ENCRYPTION_KEY")
    @classmethod
    def encryption_key_must_be_base64_fernet(cls, v: str) -> str:
        """Basic check that the encryption key looks like a Fernet key."""
        if len(v) < 20:
            raise ValueError("ENCRYPTION_KEY looks too short for a Fernet key")
        return v

    model_config = {"env_prefix": "VG_", "env_file": ".env", "env_file_encoding": "utf-8"}


# Module-level singleton — import ``settings`` to access configuration.
# The required fields are supplied via environment variables at runtime.
settings = Settings()
