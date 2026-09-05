"""Pytest configuration.

- Adds backend/ to sys.path so ``voiceguard`` resolves correctly.
- Sets the required security/database settings before any ``voiceguard``
  module is imported so the test suite runs without relying on a local
  ``.env`` or pre-set environment variables.

Note
****
The values below are **test-only** and must never be used in production.
Because ``voiceguard.config.Settings`` instantiates the module-level
``settings`` singleton at import time, these must be present before the
first ``voiceguard`` import.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure ``voiceguard.config`` resolves to backend/voiceguard/config.py
_backend_root = str(Path(__file__).resolve().parent.parent)
if _backend_root not in sys.path:
    sys.path.insert(0, _backend_root)

# Test-only settings.  ``setdefault`` lets an operator override via the
# shell while keeping the default suite self-contained.
os.environ.setdefault(
    "VG_DATABASE_URL", "postgresql+asyncpg://user:pass@localhost:5432/testdb"
)
os.environ.setdefault(
    "VG_JWT_SECRET_KEY", "test-secret-key-for-jwt-signing-operations-0000"
)
os.environ.setdefault(
    "VG_ENCRYPTION_KEY", "MDEyMzQ1Njc4OWFiY2RlZjAxMjM0NTY3ODlhYmNkZWY="
)
