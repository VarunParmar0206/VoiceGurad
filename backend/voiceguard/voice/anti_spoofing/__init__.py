"""VoiceGuard V2 — Anti-spoofing detection.

Public entry points:
  - :func:`voiceguard.voice.anti_spoofing.detect_replay` — two-tier replay
    detection (heuristic + ML classifier) [Phase 7]
  - :func:`voiceguard.voice.anti_spoofing.detect_deepfake` — two-tier
    deepfake/synthetic speech detection (heuristic + ML classifier) [Phase 8]
  - :func:`voiceguard.voice.anti_spoofing.detect_voice_conversion` —
    voice-conversion/mimicry detection (three-check heuristic) [Phase 9]
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
from voiceguard.voice.anti_spoofing.voice_conversion import (
    VoiceConversionDetectionResult,
    VoiceConversionHeuristicScores,
    detect_voice_conversion,
)
from voiceguard.voice.anti_spoofing.voice_conversion import (
    compute_heuristic_scores as compute_voice_conversion_heuristic_scores,
)

__all__ = [
    "DeepfakeDetectionResult",
    "DeepfakeHeuristicScores",
    "ReplayDetectionResult",
    "ReplayHeuristicScores",
    "VoiceConversionDetectionResult",
    "VoiceConversionHeuristicScores",
    "compute_deepfake_heuristic_scores",
    "compute_heuristic_scores",
    "compute_voice_conversion_heuristic_scores",
    "detect_deepfake",
    "detect_replay",
    "detect_voice_conversion",
]
