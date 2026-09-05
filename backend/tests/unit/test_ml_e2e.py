"""End-to-end Phase 6 test: FeatureResult -> embedding -> enroll -> verify ->
cancelable template -> registry -> training scaffold, using only synthetic data.

Verification scoring is exercised with directly-constructed L2-normalized
speaker clusters (the scaffold's random-weights model collapses synthetic
spectrograms into a single direction and cannot drive speaker separation yet);
the model + adapter are exercised for their structural contract in the same
test.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import torch

from tests.unit.ml_helpers import make_feature_result, speaker_cluster
from voiceguard.ml.models import CNNLSTMAttention
from voiceguard.ml.registry import ArtifactRegistry, ModelMetadata
from voiceguard.ml.training import TrainingConfig, run_training
from voiceguard.voice import cancelable
from voiceguard.voice.embedding import embed_result, prepare_input
from voiceguard.voice.verification import (
    EnrollmentSample,
    enroll,
    export_template,
    import_template,
    verify,
)


def _state_bytes(model: torch.nn.Module) -> bytes:
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getvalue()


def test_end_to_end_pipeline(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = CNNLSTMAttention().eval()

    # 1. Adapter + model embedding contract (structural sanity).
    prepared = prepare_input(make_feature_result(t=64, seed=2))
    embeddings = model(prepared.mel, prepared.lengths)
    assert tuple(embeddings.shape) == (1, 256)
    assert torch.isfinite(embeddings).all()

    # 2. Enrollment from an L2-normalized speaker cluster (noise-free
    #    headroom so the scaffolding can demonstrate genuine/impostor split).
    _, members, impostors = speaker_cluster(n_enroll=10, noise=0.05, seed=21)
    stats = np.arange(259)
    samples = [EnrollmentSample(m, stats) for m in members]
    profile = enroll(
        samples,
        user_id="user-7",
        embedding_dim=256,
        statistical_dim=259,
    )

    genuine = verify(profile, members[0], statistical=stats)
    intruder = verify(profile, impostors[0], statistical=stats)
    assert genuine.status == "verified"
    assert intruder.status == "rejected"
    assert genuine.score > intruder.score

    # 3. Cancelable template round trip: invariant score, no raw data.
    salt = cancelable.new_salt()
    blob = export_template(profile, password="letmein", salt=salt)
    template_profile = import_template(blob)
    transform = cancelable.derive_projection("letmein", salt, embedding_dim=256)
    from_template = verify(
        template_profile, members[0], statistical=stats, transform_r=transform
    )
    assert np.isclose(from_template.score, genuine.score, atol=1e-9)
    assert template_profile.embeddings is None

    # 4. Wrong password curve: comparisons after rotation fail.
    wrong = verify(
        template_profile,
        members[0],
        statistical=stats,
        transform_r=cancelable.derive_projection("letmeout", salt, embedding_dim=256),
    )
    assert wrong.status == "rejected"

    # 5. Registry persistence: global model plaintext, user artifact encrypted.
    registry = ArtifactRegistry(tmp_path)
    registry.save_global(
        _state_bytes(model),
        ModelMetadata(
            name="speaker_embedding_net",
            version="v1.0",
            architecture="cnn-lstm-attention",
            embedding_dim=256,
        ),
    )
    registry.save_user(
        "user-7",
        blob,
        ModelMetadata(
            name="speaker_profile",
            version="v1.0",
            architecture="cancelable",
            embedding_dim=256,
        ),
    )
    assert registry.load_global("speaker_embedding_net").data  # persisted weights
    stored = registry.load_user("user-7")
    assert stored.data == blob
    enc = (tmp_path / "users" / "user-7" / "v1.0" / "model.bin.enc").read_bytes()
    assert blob not in enc  # never at rest in plaintext

    # 6. Restored template verifies against the fresh transform.
    reconciled = verify(
        import_template(stored.data),
        members[0],
        statistical=stats,
        transform_r=transform,
    )
    assert reconciled.score == from_template.score


def test_adapter_and_model_training_smoke() -> None:
    torch.manual_seed(0)
    model = CNNLSTMAttention()
    prepared = prepare_input(make_feature_result(t=56, seed=2))
    out = model(prepared.mel, prepared.lengths)
    assert tuple(out.shape) == (1, 256)

    # consume an embedding through the public adapter too
    emb = embed_result(make_feature_result(t=48, seed=3), model)
    assert emb.shape == (256,)
    assert np.isclose(np.linalg.norm(emb), 1.0, atol=1e-4)

    cfg = TrainingConfig(seed=6, batch_size=8, epochs=1, max_frames=64)
    metrics = run_training(CNNLSTMAttention(), cfg)
    assert metrics["combined"] >= 0.0
    assert np.isfinite(metrics["combined"])
