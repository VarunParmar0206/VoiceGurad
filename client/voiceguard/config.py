"""VoiceGuard V2 — Client-side configuration.

Subset of settings relevant to the desktop Kivy client.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings


class ClientSettings(BaseSettings):
    """Client configuration loaded from environment variables."""

    # ── Application ──────────────────────────────────────────────────────
    APP_NAME: str = "VoiceGuard"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = False

    # ── Backend Connection ───────────────────────────────────────────────
    API_BASE_URL: str = Field(
        default="http://localhost:8000",
        description="Base URL of the VoiceGuard backend API.",
    )
    API_TIMEOUT_SECONDS: int = 30

    # ── Audio ────────────────────────────────────────────────────────────
    AUDIO_SAMPLE_RATE: int = 16000
    AUDIO_CHANNELS: int = 1
    AUDIO_MAX_DURATION_SECONDS: int = 30

    model_config = {"env_prefix": "VG_CLIENT_", "env_file": ".env", "env_file_encoding": "utf-8"}


client_settings = ClientSettings()
