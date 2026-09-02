"""VoiceGuard V2 — Generic async repository base.

Provides standard CRUD operations against an ``AsyncSession``.  Concrete
repositories for each entity build on this and add domain-specific
queries.
"""

from __future__ import annotations

import uuid
from typing import Generic, TypeVar

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from voiceguard.models.base import Base

ModelT = TypeVar("ModelT", bound=Base)


class RepositoryBase(Generic[ModelT]):
    """Async CRUD repository for a single ORM model."""

    model: type[ModelT]

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def add(self, entity: ModelT) -> ModelT:
        """Persist a new entity and return it with generated fields set."""
        self.session.add(entity)
        await self.session.flush()
        return entity

    async def get(self, entity_id: uuid.UUID | int) -> ModelT | None:
        """Fetch an entity by its primary key; ``None`` if not found."""
        return await self.session.get(self.model, entity_id)

    async def get_all(self) -> list[ModelT]:
        """Fetch all rows for this model."""
        result = await self.session.execute(select(self.model))
        return list(result.scalars().all())

    async def update(self, entity: ModelT) -> ModelT:
        """Mark an already-persisted entity as changed and flush."""
        await self.session.flush()
        return entity

    async def delete(self, entity: ModelT) -> None:
        """Delete an entity and flush."""
        await self.session.delete(entity)
        await self.session.flush()

    async def delete_by_id(self, entity_id: uuid.UUID | int) -> bool:
        """Delete by primary key; returns ``True`` if a row was removed."""
        entity = await self.get(entity_id)
        if entity is None:
            return False
        await self.delete(entity)
        return True
