from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import User


class UserRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_by_user_id(self, user_id: int) -> User | None:
        result = await self.session.execute(
            select(User).where(User.user_id == user_id)
        )
        return result.scalar_one_or_none()

    async def create(self, user_id: int, username: str | None, full_name: str | None) -> User:
        user = User(user_id=user_id, username=username, full_name=full_name)
        self.session.add(user)
        await self.session.commit()
        await self.session.refresh(user)
        return user

    async def get_or_create(
        self, user_id: int, username: str | None = None, full_name: str | None = None
    ) -> tuple[User, bool]:
        """Returns (user, created). created=True if new user was inserted."""
        user = await self.get_by_user_id(user_id)
        if user:
            return user, False
        user = await self.create(user_id, username, full_name)
        return user, True

    async def update_info(
        self, user_id: int, username: str | None, full_name: str | None
    ) -> None:
        await self.session.execute(
            update(User)
            .where(User.user_id == user_id)
            .values(username=username, full_name=full_name)
        )
        await self.session.commit()

    async def set_blocked(self, user_id: int, blocked: bool) -> None:
        await self.session.execute(
            update(User).where(User.user_id == user_id).values(is_blocked=blocked)
        )
        await self.session.commit()

    async def get_all_active(self) -> list[User]:
        result = await self.session.execute(
            select(User).where(User.is_active == True, User.is_blocked == False)  # noqa: E712
        )
        return list(result.scalars().all())
