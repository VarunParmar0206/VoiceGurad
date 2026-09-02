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
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 15
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    ENCRYPTION_KEY: str = Field(
        ...,
        description="Base64-encoded Fernet key for data-at-rest encryption.",
    )

    # ── Audio / Feature Extraction ───────────────────────────────────────
    AUDIO_SAMPLE_RATE: int = 16000
    AUDIO_CHANNELS: int = 1
    N_MFCC: int = 40
    N_MELS: int = 80
    HOP_LENGTH: int = 160
    N_FFT: int = 512
    F_MIN: int = 50
    F_MAX: int = 8000

    # ── Speaker Verification ─────────────────────────────────────────────
    EMBEDDING_DIM: int = 256
    EMBEDDING_INPUT_DIM: int = 256
    ENROLLMENT_MIN_SAMPLES: int = 5
    VERIFICATION_THRESHOLD: float = 0.82
    GMM_N_COMPONENTS: int = 8
    GMM_COVARIANCE_TYPE: str = "diag"
    SESSION_TIMEOUT_SECONDS: int = 900  # 15 minutes

    # ── Audio Quality ────────────────────────────────────────────────────
    MIN_AUDIO_DURATION_SECONDS: float = 0.5
    MAX_AUDIO_DURATION_SECONDS: float = 30.0
    MIN_QUALITY_SCORE: float = 0.45
    VAD_ENERGY_THRESHOLD: float = 0.02
    PRE_EMPHASIS_COEFFICIENT: float = 0.97

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
    DAILY_TRANSACTION_LIMIT: float = 50000.00
    DEFAULT_BALANCE: float = 10000.00

    # ── Model Registry ──────────────────────────────────────────────────
    MODEL_STORAGE_PATH: str = "models/"

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
settings = Settings()  # type: ignore[call-arg]
