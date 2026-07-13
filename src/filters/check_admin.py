from aiogram.filters import BaseFilter
from aiogram.types import TelegramObject

from src.core.config import get_settings


class IsAdmin(BaseFilter):
    def __init__(self):
        self.admin_ids = get_settings().ADMIN_IDS

    async def __call__(self, obj: TelegramObject) -> bool:
        return obj.from_user.id in self.admin_ids
