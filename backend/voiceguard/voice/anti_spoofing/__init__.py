"""VoiceGuard V2 — Phase 7 anti-spoofing (replay detection).

Public entry points:
  - :func:`voiceguard.voice.anti_spoofing.detect_replay` — two-tier replay
    detection (heuristic + ML classifier)
"""

from __future__ import annotations

from voiceguard.voice.anti_spoofing.replay import (
    ReplayDetectionResult,
    ReplayHeuristicScores,
    compute_heuristic_scores,
    detect_replay,
)

__all__ = [
    "ReplayDetectionResult",
    "ReplayHeuristicScores",
    "compute_heuristic_scores",
    "detect_replay",
]
