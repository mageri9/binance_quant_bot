from aiogram.fsm.state import State, StatesGroup


class FeedbackForm(StatesGroup):
    waiting_for_text = State()
    waiting_for_rating = State()
    confirm = State()
