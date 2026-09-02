"""VoiceGuard V2 — Redis-backed rate limiting.

Provides a sliding-window counter rate limiter backed by Redis.
Each check atomically increments a counter and sets a TTL.
When the counter exceeds the limit, requests are rejected with HTTP 429.

Key patterns follow Architecture 17.3:
- ``ratelimit:{scope}:{key}``  (TTL = window seconds)
"""

from __future__ import annotations

from dataclasses import dataclass

from voiceguard.db.redis import get_redis


@dataclass
class RateLimitResult:
    """Result of a rate-limit check."""

    allowed: bool
    limit: int
    remaining: int
    retry_after_seconds: int | None = None


async def check_rate_limit(
    key: str,
    limit: int,
    window_seconds: int,
) -> RateLimitResult:
    """Check and consume a rate-limit token.

    Uses Redis INCR + EXPIRE for a fixed-window counter.

    Args:
        key: Full Redis key (e.g. ``ratelimit:ip:127.0.0.1:login``).
        limit: Maximum number of requests allowed in the window.
        window_seconds: Duration of the time window in seconds.

    Returns:
        ``RateLimitResult`` indicating whether the request is allowed.
    """
    redis = get_redis()
    try:
        current = await redis.incr(key)
        if current == 1:
            await redis.expire(key, window_seconds)
        ttl = await redis.ttl(key)
    except Exception:
        # If Redis is unreachable, allow the request (fail open).
        return RateLimitResult(allowed=True, limit=limit, remaining=limit)

    remaining = max(0, limit - current)
    if current > limit:
        retry_after = ttl if ttl and ttl > 0 else window_seconds
        return RateLimitResult(
            allowed=False,
            limit=limit,
            remaining=0,
            retry_after_seconds=retry_after,
        )

    return RateLimitResult(allowed=True, limit=limit, remaining=remaining)


def make_rate_limit_key(scope: str, identifier: str) -> str:
    """Build a rate-limit Redis key.

    Args:
        scope: Rate-limit scope (e.g. ``login``, ``voice``, ``transaction``).
        identifier: Scoped identifier (e.g. IP address or user_id).

    Returns:
        Redis key string.
    """
    return f"ratelimit:{scope}:{identifier}"
