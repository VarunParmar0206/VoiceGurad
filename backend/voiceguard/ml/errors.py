"""VoiceGuard V2 — Phase 6 ML / speaker-verification errors.

A small typed hierarchy so embedding, verification, cancelable-transform,
registry, and training code fail with domain-specific exceptions instead of
leaking raw library errors.  Exception messages never include biometric
values (embeddings, features, or audio).
"""

from __future__ import annotations


class Phase6Error(Exception):
    """Base class for all Phase 6 ML / speaker-verification errors."""


class ModelError(Phase6Error):
    """The embedding model rejected its input or failed during forward."""


class EmbeddingContractError(Phase6Error):
    """A :class:`voiceguard.voice.result.FeatureResult` violated the
    Phase 5 → Phase 6 input contract (layout, dtype, sample rate, ...)."""


class EnrollmentError(Phase6Error):
    """Enrollment input was invalid (too few samples, bad shape, NaN, ...)."""


class VerificationError(Phase6Error):
    """A verification probe was malformed or scoring could not run."""


class CancelableTransformError(Phase6Error):
    """The cancelable biometric transform could not be applied/derived."""


class RegistryError(Phase6Error):
    """Model/artifact registry operation failed (missing, corrupt, mismatch)."""


class TrainingError(Phase6Error):
    """Training scaffold failed (dataset, checkpoint, or step)."""
