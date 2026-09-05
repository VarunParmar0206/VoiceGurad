"""VoiceGuard V2 — Training & evaluation scaffold.

Synthetic-only validation.  Real speaker-dataset training metrics are out of
scope for this phase; nothing here reports production performance.
"""

from __future__ import annotations

from voiceguard.ml.training.evaluate import (
    det_curve,
    eer,
    pairwise_similarity,
    tar_at_far,
    threshold_analysis,
)
from voiceguard.ml.training.losses import ArcFaceLoss, TripletLoss
from voiceguard.ml.training.train_deepfake import (
    DeepfakeTrainingConfig,
    SyntheticDeepfakeDataset,
    collate_deepfake_batch,
)
from voiceguard.ml.training.train_deepfake import (
    run_training as run_deepfake_training,
)
from voiceguard.ml.training.train_replay import (
    ReplayTrainingConfig,
    SyntheticReplayDataset,
    collate_replay_batch,
)
from voiceguard.ml.training.train_replay import (
    run_training as run_replay_training,
)
from voiceguard.ml.training.train_speaker import (
    SyntheticSpeakerDataset,
    TrainingConfig,
    collate_speaker_batch,
    deserialize_model_state,
    load_checkpoint,
    run_training,
    save_checkpoint,
    serialize_model_state,
    set_seed,
    train_one_epoch,
)

__all__ = [
    "ArcFaceLoss",
    "DeepfakeTrainingConfig",
    "ReplayTrainingConfig",
    "SyntheticDeepfakeDataset",
    "SyntheticReplayDataset",
    "SyntheticSpeakerDataset",
    "TrainingConfig",
    "TripletLoss",
    "collate_deepfake_batch",
    "collate_replay_batch",
    "collate_speaker_batch",
    "deserialize_model_state",
    "det_curve",
    "eer",
    "load_checkpoint",
    "pairwise_similarity",
    "run_deepfake_training",
    "run_replay_training",
    "run_training",
    "save_checkpoint",
    "serialize_model_state",
    "set_seed",
    "tar_at_far",
    "threshold_analysis",
    "train_one_epoch",
]
