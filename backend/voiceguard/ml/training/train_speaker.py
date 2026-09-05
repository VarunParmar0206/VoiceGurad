"""VoiceGuard V2 — Speaker-embedding training scaffold (architecture §24).

Provides reproducible trainer infrastructure:

- deterministic seeding (:func:`set_seed`);
- a synthetic speaker dataset (:class:`SyntheticSpeakerDataset`) used ONLY to
  smoke-test the pipeline — it produces meaningless embeddings and must never
  be reported as a speech model;
- :class:`TrainingConfig`, :func:`train_one_epoch`, :func:`run_training`;
- checkpoint save/load (:func:`save_checkpoint` / :func:`load_checkpoint`).

The trainer stays compatible with future real-speaker datasets (VoxCeleb1/2,
LibriSpeech): it consumes ``(mel, length, label)`` batches, and any real
dataset adapter can replace the synthetic source.  Real-dataset metrics (EER
etc.) are explicitly out of scope and never computed here.
"""

from __future__ import annotations

import io
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from voiceguard.config import settings
from voiceguard.ml.errors import TrainingError
from voiceguard.ml.training.losses import ArcFaceLoss, TripletLoss

Tensor = torch.Tensor
FloatArray = np.ndarray


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible training runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SyntheticSpeakerDataset(Dataset[tuple[FloatArray, int, int]]):
    """Deterministic synthetic speaker utterances for smoke runs.

    Each speaker has a fixed random spectral template plus per-utterance
    noise and length jitter.  This data carries **no** speech realism and is
    used only to exercise the training loop.
    """

    def __init__(
        self,
        n_speakers: int = 8,
        n_utterances: int = 6,
        n_mels: int = 80,
        max_frames: int = 96,
        min_frames: int = 32,
        seed: int = 0,
    ) -> None:
        if n_speakers < 2 or n_utterances < 2:
            raise ValueError("need >= 2 speakers and >= 2 utterances")
        rng = np.random.default_rng(seed)
        self.speakers = n_speakers
        self.utterances = n_utterances
        self.n_mels = int(n_mels)
        self.max_frames = int(max_frames)
        self.min_frames = int(min_frames)
        self.items: list[tuple[FloatArray, int, int]] = []
        for speaker in range(n_speakers):
            template = rng.standard_normal((n_mels, 1)).astype(np.float32)
            for _ in range(n_utterances):
                t_len = int(rng.integers(min_frames, max_frames + 1))
                noise_scale = float(rng.uniform(0.1, 0.5))
                noise = rng.standard_normal((n_mels, t_len)).astype(np.float32) * noise_scale
                mel = (template + noise).astype(np.float32)
                self.items.append((mel, t_len, speaker))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[FloatArray, int, int]:
        return self.items[index]


def collate_speaker_batch(
    batch: list[tuple[FloatArray, int, int]],
) -> tuple[Tensor, Tensor, Tensor]:
    """Pad a batch of mel utterances to the longest, return (mel, lengths, labels)."""
    max_len = max(item[1] for item in batch)
    n_mels = batch[0][0].shape[0]
    mels = np.zeros((len(batch), 1, n_mels, max_len), dtype=np.float32)
    lengths = np.zeros(len(batch), dtype=np.int64)
    labels = np.zeros(len(batch), dtype=np.int64)
    for i, (mel, length, label) in enumerate(batch):
        mels[i, 0, :, :length] = mel
        lengths[i] = length
        labels[i] = label
    return (
        torch.from_numpy(mels),
        torch.from_numpy(lengths),
        torch.from_numpy(labels),
    )


@dataclass
class TrainingConfig:
    """Hyperparameters for the scaffold trainer (architecture §24 defaults)."""

    seed: int = settings.SEED
    batch_size: int = 16
    epochs: int = 1  # smoke runs: single epoch; real training raises this
    lr: float = 1e-3
    weight_decay: float = 1e-5
    max_frames: int = 96
    n_speakers: int = 8
    n_utterances: int = 6
    triplet_margin: float = 0.2
    arcface_s: float = 30.0
    arcface_m: float = 0.3


def train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloader: DataLoader[tuple[Tensor, Tensor, Tensor]],
    triplet: TripletLoss,
    arcface: ArcFaceLoss,
    *,
    device: torch.device | None = None,
) -> dict[str, float]:
    """Run one training epoch and return mean scalar losses (finite)."""
    dev = device or torch.device("cpu")
    model.train()
    model.to(dev)
    total_triplet = 0.0
    total_arcface = 0.0
    total_combined = 0.0
    steps = 0
    for mel, lengths, labels in dataloader:
        mel = mel.to(dev)
        lengths = lengths.to(dev)
        labels = labels.to(dev)
        optimizer.zero_grad()
        embeddings = model(mel, lengths)
        loss_t = triplet(embeddings, labels)
        loss_a = arcface(embeddings, labels)
        loss = loss_t + loss_a
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            total_triplet += float(loss_t.detach())
            total_arcface += float(loss_a.detach())
            total_combined += float(loss.detach())
        steps += 1
    if steps == 0:
        raise TrainingError("training epoch had no batches")
    return {
        "triplet": total_triplet / steps,
        "arcface": total_arcface / steps,
        "combined": total_combined / steps,
    }


def run_training(
    model: nn.Module,
    config: TrainingConfig | None = None,
    *,
    device: torch.device | None = None,
) -> dict[str, float]:
    """Train the model on synthetic data (1-epoch smoke by default).

    Returns epoch-averaged losses.  The resulting model is NOT a useful
    speaker model — this only validates that the training machinery runs.
    """
    cfg = config or TrainingConfig()
    set_seed(cfg.seed)
    dev = device or torch.device("cpu")

    dataset = SyntheticSpeakerDataset(
        n_speakers=cfg.n_speakers,
        n_utterances=cfg.n_utterances,
        n_mels=model.n_mels if hasattr(model, "n_mels") else settings.N_MELS,
        max_frames=cfg.max_frames,
        seed=cfg.seed,
    )
    dataloader = cast(
        DataLoader[tuple[Tensor, Tensor, Tensor]],
        DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            collate_fn=collate_speaker_batch,
        ),
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, cfg.epochs)
    )
    triplet = TripletLoss(margin=cfg.triplet_margin)
    arcface = ArcFaceLoss(
        n_classes=cfg.n_speakers,
        embedding_dim=(
            model.embedding_dim
            if hasattr(model, "embedding_dim")
            else settings.EMBEDDING_DIM
        ),
        s=cfg.arcface_s,
        m=cfg.arcface_m,
    )

    summary: dict[str, float] = {}
    for _ in range(cfg.epochs):
        epoch_metrics = train_one_epoch(model, optimizer, dataloader, triplet, arcface, device=dev)
        scheduler.step()
        summary = epoch_metrics
    return summary


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    *,
    extra: dict[str, object] | None = None,
) -> None:
    """Persist model/optimizer state plus metadata to ``path``."""
    payload: dict[str, object] = {
        "epoch": int(epoch),
        "model_state": {k: v.clone() for k, v in model.state_dict().items()},
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "config": asdict(TrainingConfig()),
        "extra": dict(extra or {}),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str | Path) -> dict[str, object]:
    """Load a checkpoint dictionary (a training error indicates corruption)."""
    try:
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001 - any load failure is a checkpoint problem
        raise TrainingError(f"cannot load checkpoint {path!r}") from exc
    if not isinstance(payload, dict) or "model_state" not in payload:
        raise TrainingError(f"checkpoint {path!r} lacks model_state")
    return payload


def serialize_model_state(model: nn.Module) -> bytes:
    """Serialize a model's state_dict to bytes (for the artifact registry)."""
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getvalue()


def deserialize_model_state(data: bytes) -> dict[str, Tensor]:
    """Deserialize state_dict bytes produced by :func:`serialize_model_state`."""
    try:
        payload = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001
        raise TrainingError("cannot deserialize model state bytes") from exc
    if not isinstance(payload, dict):
        raise TrainingError("deserialized state is not a state_dict")
    return dict(payload)
