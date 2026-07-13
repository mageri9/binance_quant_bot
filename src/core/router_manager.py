from aiogram import Router

import src.handlers.user.message as user_message
import src.handlers.user.callback as user_callback
import src.handlers.admin.message as admin_message
import src.handlers.admin.callback as admin_callback


def setup_routers() -> Router:
    root = Router()

    modules = [user_message, user_callback, admin_message, admin_callback]

    for module in modules:
        module.register_handlers()
        root.include_router(module.router)

    return root
