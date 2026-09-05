"""VoiceGuard V2 — Speaker-verification loss functions (architecture §24).

- :class:`TripletLoss` — semi-hard triplet mining with margin 0.2.
- :class:`ArcFaceLoss` — additive angular-margin classification (s=30, m=0.3).

Both operate on L2-normalizable embeddings and are used by the training
scaffold.  Loss values are only meaningful for training; no synthetic loss
magnitude is ever reported as speaker-verification performance.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

Tensor = torch.Tensor
_INF = float("inf")


class TripletLoss(nn.Module):
    """Triplet loss with semi-hard negative mining.

    Per anchor the hardest same-identity embedding is the positive; the
    "best" negative is the closest different-identity embedding that lies
    within ``(d(pos), d(pos) + margin)`` (semi-hard), falling back to the
    hardest negative when no semi-hard candidate exists.  Anchors without a
    valid positive/negative pair contribute zero.
    """

    def __init__(self, margin: float = 0.2) -> None:
        super().__init__()
        if margin < 0.0:
            raise ValueError("margin must be >= 0")
        self.margin = float(margin)

    def forward(self, embeddings: Tensor, labels: Tensor) -> Tensor:
        batch = embeddings.shape[0]
        if batch < 2:
            return torch.tensor(0.0, dtype=embeddings.dtype, device=embeddings.device)

        norm = F.normalize(embeddings, p=2, dim=1)
        dist = torch.cdist(norm, norm, p=2)  # (B, B)

        same = (labels[:, None] == labels[None, :]) & ~torch.eye(
            batch, dtype=torch.bool, device=embeddings.device
        )
        different = ~(labels[:, None] == labels[None, :])

        d_pos = torch.where(same, dist, torch.full_like(dist, _INF)).min(dim=1).values
        has_pos = torch.isfinite(d_pos)

        # Hardest negative distance.
        d_neg_hard = torch.where(different, dist, torch.full_like(dist, -_INF)).max(dim=1).values
        has_neg = torch.isfinite(d_neg_hard)

        # Semi-hard candidates: negative between the positive distance and
        # positive distance + margin.
        semi_hard = different & (dist >= d_pos[:, None]) & (dist < d_pos[:, None] + self.margin)
        d_neg_semi = torch.where(semi_hard, dist, torch.full_like(dist, _INF)).min(dim=1).values
        d_neg = torch.where(torch.isfinite(d_neg_semi), d_neg_semi, d_neg_hard)

        losses = F.relu(d_pos - d_neg + self.margin)
        valid = has_pos & has_neg & torch.isfinite(losses)
        if not bool(valid.any()):
            return torch.tensor(0.0, dtype=embeddings.dtype, device=embeddings.device)
        return losses[valid].mean()


class ArcFaceLoss(nn.Module):
    """Additive angular-margin loss (Deng et al. 2019), N classes.

    ``s`` (default 30) scales the cosine logits; ``m`` (default 0.3 rad)
    is added to the angle of the target class before scaling.
    """

    def __init__(
        self,
        n_classes: int,
        embedding_dim: int = 256,
        s: float = 30.0,
        m: float = 0.3,
    ) -> None:
        super().__init__()
        if n_classes < 1 or embedding_dim < 1:
            raise ValueError("n_classes and embedding_dim must be >= 1")
        if s <= 0.0:
            raise ValueError("s must be > 0")
        if m < 0.0 or m > math.pi:
            raise ValueError("m must be within [0, pi]")
        self.s = float(s)
        self.m = float(math.cos(m))
        self.sin_m = float(math.sin(m))
        self.threshold = float(math.cos(math.pi - m))
        self.max_marginal = float(math.sin(math.pi - m) * float(m))
        self.weight = nn.Parameter(torch.empty(n_classes, embedding_dim))
        nn.init.xavier_normal_(self.weight)

    def forward(self, embeddings: Tensor, labels: Tensor) -> Tensor:
        cos = F.linear(F.normalize(embeddings), F.normalize(self.weight))  # (B, C)
        sin = torch.sqrt(torch.clamp(1.0 - cos**2, min=1e-12))

        cos_plus_m = cos * self.m - sin * self.sin_m
        # Stabilize cos(theta + m) when sin(theta + m) changes sign.
        cos_plus_m = torch.where(cos > self.threshold, cos - self.max_marginal, cos_plus_m)

        batch = embeddings.shape[0]
        rows = torch.arange(batch, device=embeddings.device)
        logits = cos.clone()
        logits[rows, labels] = cos_plus_m[rows, labels]
        logits = logits * self.s
        return F.cross_entropy(logits, labels)
