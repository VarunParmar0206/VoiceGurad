"""Tests for the artifact registry (global + per-user encrypted storage)."""

from __future__ import annotations

from pathlib import Path

import pytest

from voiceguard.ml.errors import RegistryError
from voiceguard.ml.registry import ArtifactRegistry, ModelMetadata


@pytest.fixture()
def reg(tmp_path: Path) -> ArtifactRegistry:
    return ArtifactRegistry(tmp_path)


def meta(name: str = "speaker_embedding_net", version: str = "v1.0") -> ModelMetadata:
    return ModelMetadata(
        name=name,
        version=version,
        architecture="cnn-lstm-attention",
        embedding_dim=256,
    )


def test_save_and_load_global_round_trip(reg: ArtifactRegistry) -> None:
    reg.save_global(b"weights-v1", meta(version="v1.0"))
    artifact = reg.load_global("speaker_embedding_net")
    assert artifact.data == b"weights-v1"
    assert artifact.version == "v1.0"
    assert artifact.metadata.embedding_dim == 256


def test_speaker_embedding_convenience(reg: ArtifactRegistry) -> None:
    reg.save_speaker_embedding(b"state", "v2.0")
    artifact = reg.load_global("speaker_embedding_net")
    assert artifact.version == "v2.0"
    assert artifact.data == b"state"
    assert artifact.metadata.architecture == "cnn-lstm-attention"


def test_versioning_and_promote(reg: ArtifactRegistry) -> None:
    reg.save_global(b"v1", meta(name="m", version="v1.0"))
    reg.save_global(b"v2", meta(name="m", version="v2.0"))
    assert reg.latest_version("m") == "v2.0"
    assert reg.list_versions("m") == ["v1.0", "v2.0"]
    assert reg.load_global("m", version="v1.0").data == b"v1"
    reg.promote("m", "v1.0")
    assert reg.latest_version("m") == "v1.0"


def test_previous_version_rollback(reg: ArtifactRegistry) -> None:
    reg.save_global(b"v1", meta(name="m", version="v1.0"))
    reg.save_global(b"v2", meta(name="m", version="v2.0"))
    prev = reg.previous_version("m")
    assert prev is not None
    assert prev.data == b"v1"
    assert reg.previous_version("m") is not None  # still v1 before v2

    # promote back and confirm it resolves
    reg.promote("m", "v1.0")
    assert reg.load_global("m").data == b"v1"


def test_user_artifact_encrypted_round_trip(reg: ArtifactRegistry, tmp_path: Path) -> None:
    reg.save_user("user-42", b"cancelable-template", meta(name="speaker_profile", version="v1.0"))
    stored = tmp_path / "users" / "user-42" / "v1.0"
    assert stored.exists()
    enc = stored / "model.bin.enc"
    assert enc.exists()
    raw = enc.read_bytes()
    assert b"cancelable-template" not in raw  # encrypted at rest
    artifact = reg.load_user("user-42")
    assert artifact.user_id == "user-42"
    assert artifact.data == b"cancelable-template"


def test_user_versions_and_delete(reg: ArtifactRegistry) -> None:
    reg.save_user("u1", b"t1", meta(name="speaker_profile", version="v1.0"))
    reg.save_user("u1", b"t2", meta(name="speaker_profile", version="v2.0"))
    assert reg.user_versions("u1") == ["v2.0", "v1.0"]
    assert reg.load_user("u1").data == b"t2"
    reg.delete_user("u1")
    assert reg.user_versions("u1") == []
    with pytest.raises(RegistryError):
        reg.load_user("u1")


def test_registry_rejects_empty_data(reg: ArtifactRegistry) -> None:
    with pytest.raises(RegistryError):
        reg.save_global(b"", meta())
    with pytest.raises(RegistryError):
        reg.save_user("u1", b"", meta(name="speaker_profile", version="v1.0"))


def test_check_compatible(reg: ArtifactRegistry) -> None:
    assert reg.check_compatible(meta(), {"embedding_dim": 256})
    assert not reg.check_compatible(meta(), {"embedding_dim": 512})


def test_metadata_round_trip() -> None:
    m = ModelMetadata(
        name="speaker_embedding_net",
        version="v3.0",
        architecture="cnn-lstm-attention",
        embedding_dim=256,
        description="desc",
    )
    assert ModelMetadata.from_dict(m.to_dict()) == m


def test_artifact_storage_is_under_root(reg: ArtifactRegistry, tmp_path: Path) -> None:
    reg.save_global(b"x", meta(name="m", version="v1.0"))
    assert (tmp_path / "global" / "m" / "v1.0" / "model.bin").exists()
    assert (tmp_path / "versions.json").exists() or (
        tmp_path / "global" / "m" / "versions.json"
    ).exists()
