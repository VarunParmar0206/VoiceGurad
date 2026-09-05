"""VoiceGuard V2 — Model / artifact registry (architecture §18).

Stores global models (e.g. the CNN-LSTM-Attention weights, a background GMM)
and per-user biometric artifacts under a gitignored storage root.

Layout (aligned with §18.3)
***************************
::

    {root}/global/{name}/{version}/model.bin
    {root}/global/{name}/{version}/metadata.json
    {root}/global/{name}/versions.json        (active/ordered versions)
    {root}/users/{user_id}/{version}/model.bin.enc
    {root}/users/{user_id}/{version}/metadata.json

- Global model binaries are stored as opaque bytes (weights are NOT personal
  biometric data).
- **Per-user artifacts are always encrypted at rest** with the VoiceGuard
  AES-256-GCM ``BiometricEncryptor`` and hold only cancelable/derived
  material — the registry never persists raw embeddings, raw feature vectors,
  or raw audio.
- ``versions.json`` records insertion order + the active version, enabling
  A/B coexistence (multiple versions side by side) and rollback to the
  previous version.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from voiceguard.config import settings
from voiceguard.ml.errors import RegistryError
from voiceguard.security.crypto import BiometricEncryptor

_ENCRYPTED_SUFFIX = ".bin.enc"


def _utcnow_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class ModelMetadata:
    """Non-sensitive metadata describing a stored artifact."""

    name: str
    version: str
    architecture: str
    embedding_dim: int
    description: str = ""
    created_at: str = field(default_factory=_utcnow_iso)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "architecture": self.architecture,
            "embedding_dim": self.embedding_dim,
            "description": self.description,
            "created_at": self.created_at,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelMetadata:
        try:
            return cls(
                name=str(data["name"]),
                version=str(data["version"]),
                architecture=str(data["architecture"]),
                embedding_dim=int(data["embedding_dim"]),
                description=str(data.get("description", "")),
                created_at=str(data.get("created_at", _utcnow_iso())),
                extra=dict(data.get("extra", {})),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RegistryError("malformed model metadata") from exc


@dataclass(frozen=True)
class GlobalArtifact:
    """A stored global model plus its metadata."""

    name: str
    version: str
    data: bytes
    metadata: ModelMetadata


@dataclass(frozen=True)
class UserArtifact:
    """A decrypted per-user biometric artifact plus its metadata."""

    user_id: str
    version: str
    data: bytes
    metadata: ModelMetadata


class ArtifactRegistry:
    """Local filesystem model/artifact registry.

    Parameters
    ----------
    storage_path : str | Path
        Root directory (default: ``settings.MODEL_STORAGE_PATH`` = ``models/``).
    encryptor : BiometricEncryptor | None
        AES-256-GCM encryptor for per-user artifacts.  When omitted one is
        built from ``settings.ENCRYPTION_KEY`` (must be configured).
    """

    def __init__(
        self,
        storage_path: str | Path | None = None,
        encryptor: BiometricEncryptor | None = None,
    ) -> None:
        root = settings.MODEL_STORAGE_PATH if storage_path is None else storage_path
        self.root = Path(root)
        self.encryptor = encryptor or BiometricEncryptor(settings.ENCRYPTION_KEY)

    def check_compatible(self, metadata: ModelMetadata, required: dict[str, Any]) -> bool:
        """Return True if a metadata object satisfies every ``required`` key."""
        return all(
            getattr(metadata, key, None) == value for key, value in required.items()
        )

    # ── Global models ───────────────────────────────────────────────────

    def save_global(self, data: bytes, metadata: ModelMetadata) -> None:
        """Persist a global model under ``{name}/{version}`` and register it."""
        if not isinstance(data, bytes) or not data:
            raise RegistryError("model data must be non-empty bytes")
        version = self._global_version_dir(metadata.name, metadata.version)
        self._atomic_write(version / "model.bin", data)
        self._atomic_write(version / "metadata.json", _json_bytes(metadata.to_dict()))
        self._register_version(metadata.name, metadata.version)

    def load_global(self, name: str, version: str | None = None) -> GlobalArtifact:
        """Load a global model; ``version`` defaults to the active one."""
        resolved = self._resolve_global_version(name, version)
        data = self._read(resolved / "model.bin")
        meta = self._read_metadata(resolved)
        return GlobalArtifact(name=meta.name, version=meta.version, data=data, metadata=meta)

    def save_speaker_embedding(
        self,
        state_bytes: bytes,
        version: str,
        *,
        embedding_dim: int = 256,
        description: str = "",
    ) -> None:
        """Convenience wrapper storing the global CNN-LSTM-Attention weights."""
        metadata = ModelMetadata(
            name="speaker_embedding_net",
            version=version,
            architecture="cnn-lstm-attention",
            embedding_dim=embedding_dim,
            description=description,
        )
        self.save_global(state_bytes, metadata)

    def latest_version(self, name: str) -> str | None:
        """Active version for a global model, or ``None`` if unregistered."""
        versions = self._load_versions(name)
        return versions.get("active")

    def list_versions(self, name: str) -> list[str]:
        """All registered versions for a global model (insertion order)."""
        versions = self._load_versions(name)
        return list(versions.get("all", []))

    def previous_version(self, name: str) -> GlobalArtifact | None:
        """Load the version immediately before the active one, if any."""
        versions = self._load_versions(name)
        order = list(versions.get("all", []))
        active = versions.get("active")
        if active in order:
            idx = order.index(active)
            if idx > 0:
                return self.load_global(name, order[idx - 1])
        return None

    def promote(self, name: str, version: str) -> None:
        """Set the active version (A/B coexistence + rollback support)."""
        versions = self._load_versions(name)
        order = list(versions.get("all", []))
        if version not in order:
            raise RegistryError(f"version {version!r} is not registered for {name!r}")
        versions["active"] = version
        self._atomic_write(self._global_dir(name) / "versions.json", _json_bytes(versions))

    # ── Per-user artifacts (AES-256-GCM encrypted at rest) ──────────────

    def save_user(
        self,
        user_id: str,
        data: bytes,
        metadata: ModelMetadata,
        *,
        encrypt: bool = True,
    ) -> None:
        """Persist a per-user artifact, encrypted at rest by default.

        ``metadata`` must not contain biometric material (it is stored in
        plaintext JSON); the payload bytes are always encrypted.
        """
        if not user_id:
            raise RegistryError("user_id must be non-empty")
        if not isinstance(data, bytes) or not data:
            raise RegistryError("user artifact data must be non-empty bytes")
        directory = self._user_version_dir(user_id, metadata.version)
        payload = self.encryptor.encrypt(data) if encrypt else data
        suffix = _ENCRYPTED_SUFFIX if encrypt else ".bin"
        self._atomic_write(directory / f"model{suffix}", payload)
        self._atomic_write(directory / "metadata.json", _json_bytes(metadata.to_dict()))

    def load_user(self, user_id: str, version: str | None = None) -> UserArtifact:
        """Load and decrypt a per-user artifact (latest version by default)."""
        root = self._user_dir(user_id)
        if not root.exists():
            raise RegistryError(f"no stored artifacts for user {user_id!r}")
        versions = sorted(
            (p.name for p in root.iterdir() if p.is_dir()),
            reverse=True,
        )
        if not versions:
            raise RegistryError(f"no stored artifacts for user {user_id!r}")
        resolved = version if version is not None else versions[0]
        directory = self._user_version_dir(user_id, resolved)
        if not directory.exists():
            raise RegistryError(
                f"user {user_id!r} has no artifact version {resolved!r}"
            )
        meta = self._read_metadata(directory)
        secret_path = directory / f"model{_ENCRYPTED_SUFFIX}"
        plain: bytes
        if secret_path.exists():
            plain = self.encryptor.decrypt(self._read(secret_path))
        else:
            plain = self._read(directory / "model.bin")
        return UserArtifact(user_id=user_id, version=resolved, data=plain, metadata=meta)

    def delete_user(self, user_id: str) -> None:
        """Remove all versions of a user's artifacts (revocation)."""
        directory = self._user_dir(user_id)
        if directory.exists():
            self._remove_tree(directory)

    def user_versions(self, user_id: str) -> list[str]:
        directory = self._user_dir(user_id)
        if not directory.exists():
            return []
        return sorted((p.name for p in directory.iterdir() if p.is_dir()), reverse=True)

    # ── Internals ───────────────────────────────────────────────────────

    def _global_dir(self, name: str) -> Path:
        return self.root / "global" / name

    def _global_version_dir(self, name: str, version: str) -> Path:
        self._validate_version(version)
        path = self._global_dir(name) / version
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _user_dir(self, user_id: str) -> Path:
        return self.root / "users" / user_id

    def _user_version_dir(self, user_id: str, version: str) -> Path:
        self._validate_version(version)
        path = self._user_dir(user_id) / version
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _resolve_global_version(self, name: str, version: str | None) -> Path:
        directory = self._global_dir(name)
        if not directory.exists():
            raise RegistryError(f"no global model registered as {name!r}")
        if version is not None:
            self._validate_version(version)
            path = directory / version
            if not path.exists():
                raise RegistryError(f"version {version!r} of {name!r} does not exist")
            return path
        active = self._load_versions(name).get("active")
        if active is None:
            raise RegistryError(f"no active version for global model {name!r}")
        path = directory / str(active)
        if not path.exists():
            raise RegistryError(f"active version {active!r} of {name!r} is missing")
        return path

    def _register_version(self, name: str, version: str) -> None:
        versions = self._load_versions(name)
        order = list(versions.get("all", []))
        if version not in order:
            order.append(version)
        # The most recently saved version becomes the active one; callers can
        # override with promote() for A/B coexistence or rollback.
        versions["active"] = version
        versions["all"] = order
        directory = self._global_dir(name)
        directory.mkdir(parents=True, exist_ok=True)
        self._atomic_write(directory / "versions.json", _json_bytes(versions))

    def _load_versions(self, name: str) -> dict[str, Any]:
        path = self._global_dir(name) / "versions.json"
        if not path.exists():
            return {"all": [], "active": None}
        try:
            return cast(dict[str, Any], json.loads(self._read(path)))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise RegistryError(f"corrupt versions.json for {name!r}") from exc

    def _read_metadata(self, directory: Path) -> ModelMetadata:
        path = directory / "metadata.json"
        if not path.exists():
            raise RegistryError(f"missing metadata.json in {directory}")
        try:
            data = json.loads(self._read(path))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise RegistryError(f"corrupt metadata.json in {directory}") from exc
        return ModelMetadata.from_dict(data)

    @staticmethod
    def _validate_version(version: str) -> None:
        if not version or any(ch in version for ch in ("/", "\\", "..")):
            raise RegistryError("invalid artifact version identifier")

    @staticmethod
    def _read(path: Path) -> bytes:
        try:
            return path.read_bytes()
        except OSError as exc:
            raise RegistryError(f"cannot read {path}") from exc

    @staticmethod
    def _atomic_write(path: Path, payload: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, path)
        except OSError as exc:
            with contextlib.suppress(OSError):
                os.unlink(tmp_name)
            raise RegistryError(f"cannot write {path}") from exc

    @staticmethod
    def _remove_tree(path: Path) -> None:
        for child in sorted(path.rglob("*"), reverse=True):
            if child.is_dir():
                child.rmdir()
            else:
                child.unlink()
        path.rmdir()


def _json_bytes(data: dict[str, Any]) -> bytes:
    try:
        return json.dumps(data, sort_keys=True).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RegistryError("metadata contains non-JSON-serializable values") from exc
