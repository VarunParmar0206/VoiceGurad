"""VoiceGuard V2 — Phase 8 deepfake-detector training scaffold.

Synthetic-only validation.  Real deepfake-detection training requires
ASVspoof 2021 logical access and Wavefake datasets; the training pipeline
is compatible with that data once it becomes available.  No production
AUC-ROC, ACER, BPCER, or APCER is claimed from synthetic runs.

Disclaimers
***********
All metrics produced from synthetic data are **test-only artifacts** and
must never be reported as VoiceGuard anti-spoofing performance.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from voiceguard.ml.errors import TrainingError
from voiceguard.ml.models.anti_spoofing_deepfake import DeepfakeDetector

FloatArray = npt.NDArray[np.float32]
Tensor = torch.Tensor


# ── Synthetic dataset ────────────────────────────────────────────────


class SyntheticDeepfakeDataset(Dataset[tuple[Tensor, Tensor]]):
    """Deterministic synthetic bona fide/deepfake mel-spectrogram pairs.

    Each sample is a random mel-spectrogram with a binary label:
    - label 0 (bona fide): random spectral template + natural variation
    - label 1 (synthetic): template with spectral discontinuities +
      unnatural periodicity + flattened cepstral variance

    This data carries **no** speech realism and is used only to exercise
    the training loop.
    """

    def __init__(
        self,
        n_samples: int = 256,
        n_mels: int = 80,
        max_frames: int = 300,
        *,
        seed: int = 42,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.items: list[tuple[Tensor, Tensor]] = []
        for _ in range(n_samples):
            label = int(rng.integers(0, 2))
            t_len = int(rng.integers(50, max_frames + 1))
            template = rng.standard_normal((n_mels, 1)).astype(np.float32)
            noise = rng.standard_normal((n_mels, t_len)).astype(np.float32) * 0.2
            mel = template + noise
            if label == 1:
                # Simulate deepfake: spectral discontinuities at frame
                # boundaries (neural vocoder artifact).
                boundary_mask = np.zeros((n_mels, t_len), dtype=np.float32)
                for i in range(0, t_len, 8):
                    if i + 1 < t_len:
                        boundary_mask[:, i] = rng.uniform(
                            0.3, 0.8, n_mels
                        ).astype(np.float32)
                mel = mel + boundary_mask
                # Unnatural periodicity: repeat a short spectral pattern.
                period = int(rng.integers(4, 9))
                for start in range(period, t_len, period):
                    end = min(start + period, t_len)
                    mel[:, start:end] = mel[:, 0 : end - start]
                # Flatten cepstral variance (make spectrum more uniform).
                spectral_std = np.std(mel, axis=0, keepdims=True)
                mel = mel / np.maximum(spectral_std, 1e-6) * 0.5
            mel = mel.astype(np.float32)
            # Pad to max_frames.
            padded = np.zeros((n_mels, max_frames), dtype=np.float32)
            padded[:, :t_len] = mel
            self.items.append((
                torch.from_numpy(padded[np.newaxis, :, :]),  # (1, n_mels, T)
                torch.tensor([label], dtype=torch.float32),
            ))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return self.items[index]


def collate_deepfake_batch(
    batch: list[tuple[Tensor, Tensor]],
) -> tuple[Tensor, Tensor]:
    """Stack mel-spectrograms and labels into a batch."""
    mels = torch.stack([item[0] for item in batch])  # (B, 1, n_mels, T)
    labels = torch.stack([item[1] for item in batch])  # (B, 1)
    return mels, labels


# ── Training config ──────────────────────────────────────────────────


@dataclass
class DeepfakeTrainingConfig:
    """Configuration for deepfake-detector smoke training."""

    seed: int = 42
    batch_size: int = 16
    epochs: int = 1
    lr: float = 1e-3
    weight_decay: float = 1e-5
    n_mels: int = 80
    max_frames: int = 300
    n_samples: int = 256
    cnn_channels: tuple[int, int, int] = (32, 64, 128)
    dropout: float = 0.3


# ── Training loop ────────────────────────────────────────────────────


def set_seed(seed: int) -> None:
    """Set deterministic seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloader: DataLoader[tuple[Tensor, Tensor]],
    *,
    device: torch.device | None = None,
) -> dict[str, float]:
    """Run one training epoch and return mean scalar losses (finite)."""
    dev = device or torch.device("cpu")
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    n_batches = 0

    for mels, labels in dataloader:
        mels = mels.to(dev)
        labels = labels.to(dev)
        optimizer.zero_grad()
        logits, _ = model(mels)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        n_batches += 1

    if n_batches == 0:
        raise TrainingError("no batches processed during training")
    return {"bce_loss": total_loss / n_batches}


def run_training(
    model: nn.Module | None = None,
    config: DeepfakeTrainingConfig | None = None,
    *,
    device: torch.device | None = None,
) -> dict[str, float]:
    """Train the deepfake detector on synthetic data (1-epoch smoke).

    Returns epoch-averaged losses.  The resulting model is NOT a useful
    deepfake detector — this only validates that the training machinery runs.
    """
    cfg = config or DeepfakeTrainingConfig()
    set_seed(cfg.seed)
    dev = device or torch.device("cpu")

    if model is None:
        model = DeepfakeDetector(
            n_mels=cfg.n_mels,
            cnn_channels=cfg.cnn_channels,
            dropout=cfg.dropout,
        )

    dataset = SyntheticDeepfakeDataset(
        n_samples=cfg.n_samples,
        n_mels=cfg.n_mels,
        max_frames=cfg.max_frames,
        seed=cfg.seed,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_deepfake_batch,
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    summary: dict[str, float] = {}
    for _ in range(cfg.epochs):
        epoch_metrics = train_one_epoch(model, optimizer, dataloader, device=dev)
        summary = epoch_metrics
    return summary


# ── Checkpoint utilities ─────────────────────────────────────────────


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    *,
    extra: dict[str, object] | None = None,
) -> None:
    """Persist model/optimizer state plus metadata to ``path``."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    state: dict[str, object] = {
        "model_state": model.state_dict(),
        "epoch": epoch,
    }
    if optimizer is not None:
        state["optimizer_state"] = optimizer.state_dict()
    if extra:
        state.update(extra)
    torch.save(state, p)


def load_checkpoint(path: str | Path) -> dict[str, object]:
    """Load a checkpoint dictionary (a training error indicates corruption)."""
    try:
        payload = torch.load(
            path, map_location="cpu", weights_only=True
        )
    except Exception as exc:  # noqa: BLE001
        raise TrainingError(f"cannot load checkpoint {path!r}") from exc
    if not isinstance(payload, dict) or "model_state" not in payload:
        raise TrainingError(f"checkpoint {path!r} lacks model_state")
    return payload


def serialize_model_state(model: nn.Module) -> bytes:
    """Serialize a model's state_dict to bytes."""
    import io

    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getvalue()


def deserialize_model_state(data: bytes) -> dict[str, Tensor]:
    """Deserialize state_dict bytes produced by :func:`serialize_model_state`."""
    import io

    try:
        payload = torch.load(
            io.BytesIO(data), map_location="cpu", weights_only=True
        )
    except Exception as exc:  # noqa: BLE001
        raise TrainingError("cannot deserialize model state bytes") from exc
    if not isinstance(payload, dict):
        raise TrainingError("deserialized state is not a state_dict")
    return dict(payload)
