"""VoiceGuard V2 — Redis connection manager.

Provides async access to Redis for session caching, rate limiting,
failed-attempt tracking, and challenge nonces (see Architecture §17.3).

Usage::

    from voiceguard.db.redis import get_redis

    redis = get_redis()
    await redis.set("challenge:{id}", "value", ex=30)

Keys follow the patterns defined in the architecture:
- ``session:{user_id}:{session_id}``   (TTL 15 min)
- ``ratelimit:{user_id}:voice``        (TTL 1 hour)
- ``ratelimit:{user_id}:transaction``  (TTL 1 hour)
- ``ratelimit:{ip}:login``             (TTL 15 min)
- ``attempts:{user_id}:failed``        (TTL 30 min)
- ``tx-history:{user_id}``             (TTL 60 sec)
- ``challenge:{challenge_id}``         (TTL 30 sec)
"""

from __future__ import annotations

from redis.asyncio import Redis

from voiceguard.config import settings


def create_redis() -> Redis:
    """Create a new async Redis client from the configured URL."""
    redis = Redis.from_url(
        settings.REDIS_URL,
        encoding="utf-8",
        decode_responses=True,
        socket_timeout=5,
    )
    return redis


async def ping_redis(redis: Redis | None = None) -> bool:
    """Return ``True`` if the Redis server responds to PING."""
    client = redis or get_redis()
    try:
        return bool(await client.ping())
    except Exception:
        return False


async def close_redis(redis: Redis | None = None) -> None:
    """Close the Redis connection pool."""
    client = redis or get_redis()
    await client.aclose()


# Module-level singleton — started lazily on first use.
_redis: Redis | None = None


def get_redis() -> Redis:
    """Return the shared Redis client, creating it on first use."""
    global _redis
    if _redis is None:
        _redis = create_redis()
    return _redis
