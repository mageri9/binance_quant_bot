import time
from typing import Any, Awaitable, Callable

from aiogram import BaseMiddleware
from aiogram.types import Message, TelegramObject
from loguru import logger
from redis.asyncio import Redis

from src.core.config import get_settings

settings = get_settings()

RATE_LIMIT_SCRIPT = """
local key = KEYS[1]
local now = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local limit = tonumber(ARGV[3])

-- Remove timestamps outside the window
redis.call('ZREMRANGEBYSCORE', key, '-inf', now - window * 1000)

local count = redis.call('ZCARD', key)
if count < limit then
    redis.call('ZADD', key, now, now)
    redis.call('PEXPIRE', key, window * 1000)
    return 1
else
    return 0
end
"""


class RateLimitMiddleware(BaseMiddleware):
    """
    Sliding window rate limiter backed by Redis.
    Allows `limit` requests per `period` seconds per user.
    Admins are exempt.
    """

    def __init__(self, redis: Redis, limit: int | None = None, period: int | None = None):
        self.redis = redis
        self.limit = limit or settings.RATE_LIMIT_CALLS
        self.period = period or settings.RATE_LIMIT_PERIOD
        self._script = redis.register_script(RATE_LIMIT_SCRIPT)
        super().__init__()

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        user = data.get("event_from_user")
        if not user:
            return await handler(event, data)

        # Admins bypass rate limiting
        if user.id in settings.ADMIN_IDS:
            return await handler(event, data)

        key = f"rate_limit:{user.id}"
        now_ms = int(time.time() * 1000)

        allowed = await self._script(
            keys=[key],
            args=[now_ms, self.period, self.limit],
        )

        if not allowed:
            logger.warning(f"Rate limit hit: user_id={user.id} ({self.limit} req/{self.period}s)")
            if isinstance(event, Message):
                await event.answer(
                    f"⚠️ Слишком много запросов. Подождите {self.period} секунд."
                )
            return None

        return await handler(event, data)
