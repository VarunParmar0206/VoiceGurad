"""VoiceGuard V2 — Repository layer.

Concrete repositories for each domain entity.  All repositories are
constructed with an ``AsyncSession`` and expose async CRUD operations.
"""

from __future__ import annotations

from voiceguard.repositories.audit_repository import AuditLogRepository
from voiceguard.repositories.auth_attempt_repository import AuthAttemptRepository
from voiceguard.repositories.base import RepositoryBase
from voiceguard.repositories.challenge_repository import ChallengeRepository
from voiceguard.repositories.session_repository import SessionRepository
from voiceguard.repositories.transaction_repository import TransactionRepository
from voiceguard.repositories.user_repository import UserRepository
from voiceguard.repositories.voice_repository import (
    VoiceModelRepository,
    VoiceTemplateRepository,
)

__all__ = [
    "RepositoryBase",
    "UserRepository",
    "VoiceTemplateRepository",
    "VoiceModelRepository",
    "SessionRepository",
    "TransactionRepository",
    "AuditLogRepository",
    "ChallengeRepository",
    "AuthAttemptRepository",
]
