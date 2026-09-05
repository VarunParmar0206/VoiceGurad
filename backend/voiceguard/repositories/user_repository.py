"""VoiceGuard V2 — User repository."""

from __future__ import annotations

from sqlalchemy import select

from voiceguard.models.user import User
from voiceguard.repositories.base import RepositoryBase


class UserRepository(RepositoryBase[User]):
    """Data-access operations for the ``users`` table."""

    model = User

    async def get_by_username(self, username: str) -> User | None:
        result = await self.session.execute(
            select(User).where(User.username == username)
        )
        return result.scalar_one_or_none()

    async def get_by_email(self, email: str) -> User | None:
        result = await self.session.execute(
            select(User).where(User.email == email)
        )
        return result.scalar_one_or_none()

    async def exists_username(self, username: str) -> bool:
        result = await self.session.execute(
            select(User.id).where(User.username == username)
        )
        return result.scalar_one_or_none() is not None

    async def exists_email(self, email: str) -> bool:
        result = await self.session.execute(
            select(User.id).where(User.email == email)
        )
        return result.scalar_one_or_none() is not None

    async def set_totp_secret(self, user: User, encrypted_secret: bytes) -> None:
        """Persist an encrypted TOTP secret for a user."""
        user.totp_secret = encrypted_secret
        await self.session.flush()

    async def set_totp_enabled(self, user: User, enabled: bool) -> None:
        """Enable or disable TOTP for a user."""
        user.totp_enabled = enabled
        await self.session.flush()

    async def set_locked(self, user: User, locked: bool) -> None:
        """Set the account-lockout flag."""
        user.is_locked = locked
        await self.session.flush()
