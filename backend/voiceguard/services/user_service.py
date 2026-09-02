"""VoiceGuard V2 — User service.

Business logic for user management.  Phase 3 provides a minimal
service that wires repositories to route handlers.  Password hashing
and authentication logic belong to Phase 4.
"""

from __future__ import annotations

import uuid
from decimal import Decimal

from sqlalchemy.ext.asyncio import AsyncSession

from voiceguard.config import settings
from voiceguard.models.user import User
from voiceguard.repositories import UserRepository


class UserService:
    """Encapsulates user-related business logic."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._repo = UserRepository(session)

    async def get_by_id(self, user_id: uuid.UUID) -> User | None:
        return await self._repo.get(user_id)

    async def get_by_username(self, username: str) -> User | None:
        return await self._repo.get_by_username(username)

    async def get_by_email(self, email: str) -> User | None:
        return await self._repo.get_by_email(email)

    async def username_exists(self, username: str) -> bool:
        return await self._repo.exists_username(username)

    async def email_exists(self, email: str) -> bool:
        return await self._repo.exists_email(email)

    async def create_user(
        self,
        username: str,
        email: str,
        password_hash: str,
        display_name: str | None = None,
    ) -> User:
        """Create a new user with a default balance.

        ``password_hash`` is expected to be pre-hashed by the caller
        (Phase 4 will use Argon2id).
        """
        user = User(
            id=uuid.uuid4(),
            username=username,
            email=email,
            password_hash=password_hash,
            display_name=display_name,
            is_active=True,
            is_locked=False,
            balance=Decimal(str(settings.DEFAULT_BALANCE)),
            daily_limit=Decimal(str(settings.DAILY_TRANSACTION_LIMIT)),
        )
        return await self._repo.add(user)

    async def update_profile(
        self,
        user: User,
        *,
        display_name: str | None = None,
        email: str | None = None,
    ) -> User:
        if display_name is not None:
            user.display_name = display_name
        if email is not None and email != user.email:
            existing = await self._repo.get_by_email(email)
            if existing and existing.id != user.id:
                raise ValueError("Email already in use")
            user.email = email
        await self._repo.update(user)
        return user

    async def update_password(
        self,
        user: User,
        new_password_hash: str,
    ) -> User:
        user.password_hash = new_password_hash
        await self._repo.update(user)
        return user

    async def get_balance(self, user: User) -> Decimal:
        return user.balance or Decimal("0.00")

    async def debit_balance(
        self,
        user: User,
        amount: Decimal,
    ) -> None:
        """Atomically deduct amount from user balance.

        Raises ValueError if insufficient balance.
        """
        if user.balance < amount:
            raise ValueError("Insufficient balance")
        user.balance = user.balance - amount
        await self._repo.update(user)

    async def credit_balance(
        self,
        user: User,
        amount: Decimal,
    ) -> None:
        user.balance = user.balance + amount
        await self._repo.update(user)
