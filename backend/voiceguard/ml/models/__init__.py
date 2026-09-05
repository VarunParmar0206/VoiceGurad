"""VoiceGuard V2 — ML model definitions."""

from __future__ import annotations

from voiceguard.ml.models.anti_spoofing_deepfake import DeepfakeDetector
from voiceguard.ml.models.anti_spoofing_replay import ReplayDetector
from voiceguard.ml.models.cnn_lstm_attention import CNNLSTMAttention, pool_length

__all__ = ["CNNLSTMAttention", "DeepfakeDetector", "ReplayDetector", "pool_length"]
