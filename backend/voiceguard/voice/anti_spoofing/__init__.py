"""VoiceGuard V2 — Anti-spoofing detection.

Public entry points:
  - :func:`voiceguard.voice.anti_spoofing.detect_replay` — two-tier replay
    detection (heuristic + ML classifier) [Phase 7]
  - :func:`voiceguard.voice.anti_spoofing.detect_deepfake` — two-tier
    deepfake/synthetic speech detection (heuristic + ML classifier) [Phase 8]
"""

from __future__ import annotations

from voiceguard.voice.anti_spoofing.deepfake import (
    DeepfakeDetectionResult,
    DeepfakeHeuristicScores,
    detect_deepfake,
)
from voiceguard.voice.anti_spoofing.deepfake import (
    compute_heuristic_scores as compute_deepfake_heuristic_scores,
)
from voiceguard.voice.anti_spoofing.replay import (
    ReplayDetectionResult,
    ReplayHeuristicScores,
    compute_heuristic_scores,
    detect_replay,
)

__all__ = [
    "DeepfakeDetectionResult",
    "DeepfakeHeuristicScores",
    "ReplayDetectionResult",
    "ReplayHeuristicScores",
    "compute_deepfake_heuristic_scores",
    "compute_heuristic_scores",
    "detect_deepfake",
    "detect_replay",
]
