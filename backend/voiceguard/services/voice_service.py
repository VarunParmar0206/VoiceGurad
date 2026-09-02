"""VoiceGuard V2 — Voice service.

Business logic for voice enrollment and verification.
Phase 3 provides stubs that wire the route layer to the service layer.
Actual voice processing (preprocessing, feature extraction, ML inference)
belongs to Phases 5-6.
"""

from __future__ import annotations

import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from voiceguard.repositories import (
    VoiceModelRepository,
    VoiceTemplateRepository,
)


class VoiceService:
    """Encapsulates voice-related business logic.

    **Phase 3 stubs:** enrollment and verification endpoints return
    placeholder responses.  The voice pipeline will be wired in
    Phases 5-6 (see Architecture 4.2).
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._template_repo = VoiceTemplateRepository(session)
        self._model_repo = VoiceModelRepository(session)

    async def enroll_user(
        self,
        user_id: uuid.UUID,
    ) -> dict[str, object]:
        """Enroll a user with voice samples.

        Phase 3 stub: returns a placeholder response.
        Phase 6 will implement full enrollment (feature extraction,
        embedding generation, cancelable transform, GMM training).
        """
        return {
            "status": "enrolled",
            "user_id": user_id,
            "template_id": uuid.uuid4(),
            "enrollment_samples": 0,
        }

    async def get_enrollment_status(
        self,
        user_id: uuid.UUID,
    ) -> dict[str, object]:
        """Check enrollment status for a user.

        Phase 3 stub: always reports not enrolled.
        """
        return {
            "user_id": user_id,
            "is_enrolled": False,
            "enrollment_samples": 0,
            "model_version": None,
            "created_at": None,
        }

    async def verify_voice(
        self,
        user_id: uuid.UUID,
    ) -> dict[str, object]:
        """Verify a voice sample against the user's enrollment.

        Phase 3 stub: returns a placeholder rejection.
        Phase 6 will implement the full verification pipeline
        (preprocessing, feature extraction, embedding, anti-spoof,
        scoring — see Architecture 4.2).
        """
        return {
            "status": "rejected",
            "voice_score": 0.0,
            "anti_spoof_score": None,
            "threshold": 0.0,
            "reason": "voice_pipeline_not_implemented",
        }

    async def re_enroll_user(
        self,
        user_id: uuid.UUID,
    ) -> dict[str, object]:
        """Re-enroll a user (replace existing template).

        Phase 3 stub: returns placeholder response.
        """
        await self._template_repo.deactivate_for_user(user_id)
        return await self.enroll_user(user_id)
