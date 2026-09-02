"""Pytest configuration — adds backend/ to sys.path for imports."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure ``voiceguard.config`` resolves to backend/voiceguard/config.py
_backend_root = str(Path(__file__).resolve().parent.parent)
if _backend_root not in sys.path:
    sys.path.insert(0, _backend_root)
