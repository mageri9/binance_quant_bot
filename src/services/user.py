import json
from typing import Any

from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from src.crud.user import UserRepository
from src.db.models import User

USER_CACHE_TTL = 300  # 5 minutes


def _user_cache_key(user_id: int) -> str:
    return f"user:{user_id}"


def _user_to_dict(user: User) -> dict[str, Any]:
    return {
        "id": user.id,
        "user_id": user.user_id,
        "username": user.username,
        "full_name": user.full_name,
        "is_active": user.is_active,
        "is_blocked": user.is_blocked,
    }


class UserService:
    def __init__(self, session: AsyncSession, redis: Redis):
        self.repo = UserRepository(session)
        self.redis = redis

    async def _invalidate_cache(self, user_id: int) -> None:
        await self.redis.delete(_user_cache_key(user_id))

    async def get_cached(self, user_id: int) -> dict | None:
        raw = await self.redis.get(_user_cache_key(user_id))
        if raw:
            return json.loads(raw)
        return None

    async def register_or_update(
        self,
        user_id: int,
        username: str | None = None,
        full_name: str | None = None,
    ) -> tuple[User, bool]:
        """
        Upserts user. Updates username/full_name on each call (they may change).
        Returns (user, is_new).
        """
        user, created = await self.repo.get_or_create(user_id, username, full_name)

        if not created:
            await self.repo.update_info(user_id, username, full_name)

        await self.redis.setex(
            _user_cache_key(user_id),
            USER_CACHE_TTL,
            json.dumps(_user_to_dict(user)),
        )
        return user, created
