"""VoiceGuard V2 — Voice template and voice model repositories."""

from __future__ import annotations

import uuid

from sqlalchemy import select

from voiceguard.models.voice_model import VoiceModel, VoiceTemplate
from voiceguard.repositories.base import RepositoryBase


class VoiceTemplateRepository(RepositoryBase[VoiceTemplate]):
    """Data-access operations for the ``voice_templates`` table."""

    model = VoiceTemplate

    async def get_active_for_user(self, user_id: uuid.UUID) -> VoiceTemplate | None:
        """Return the most recent active template for a user, if any."""
        result = await self.session.execute(
            select(VoiceTemplate)
            .where(VoiceTemplate.user_id == user_id, VoiceTemplate.is_active.is_(True))
            .order_by(VoiceTemplate.created_at.desc())
        )
        return result.scalars().first()

    async def list_for_user(self, user_id: uuid.UUID) -> list[VoiceTemplate]:
        result = await self.session.execute(
            select(VoiceTemplate)
            .where(VoiceTemplate.user_id == user_id)
            .order_by(VoiceTemplate.created_at.desc())
        )
        return list(result.scalars().all())

    async def deactivate_for_user(self, user_id: uuid.UUID) -> None:
        """Set ``is_active = False`` for all of a user's templates.

        Called before inserting a new template during re-enrollment.
        """
        result = await self.session.execute(
            select(VoiceTemplate).where(VoiceTemplate.user_id == user_id)
        )
        for template in result.scalars().all():
            template.is_active = False
        await self.session.flush()


class VoiceModelRepository(RepositoryBase[VoiceModel]):
    """Data-access operations for the ``voice_models`` table."""

    model = VoiceModel

    async def get_for_user(self, user_id: uuid.UUID, model_type: str) -> VoiceModel | None:
        """Fetch the most recent model of a given type for a user."""
        result = await self.session.execute(
            select(VoiceModel)
            .where(VoiceModel.user_id == user_id, VoiceModel.model_type == model_type)
            .order_by(VoiceModel.created_at.desc())
        )
        return result.scalars().first()

    async def list_for_user(self, user_id: uuid.UUID) -> list[VoiceModel]:
        result = await self.session.execute(
            select(VoiceModel)
            .where(VoiceModel.user_id == user_id)
            .order_by(VoiceModel.created_at.desc())
        )
        return list(result.scalars().all())

    async def list_for_template(self, template_id: uuid.UUID) -> list[VoiceModel]:
        result = await self.session.execute(
            select(VoiceModel).where(VoiceModel.template_id == template_id)
        )
        return list(result.scalars().all())
