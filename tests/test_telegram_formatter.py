from decimal import Decimal

from src.telegram.formatter import TradingNotification, format_trading_notification


def test_open_notification_is_user_facing_and_not_engine_log():
    message = format_trading_notification(TradingNotification(
        kind="position_opened", symbol="ETH/USDT", side="buy",
        amount=Decimal("0.256"), price=Decimal("1931.07"), order_id="184736251",
        model_id="eth-1h@20260723", confidence=0.68, sl_ok=True, tp_ok=True,
    ))

    assert "ETH/USDT" in message
    assert "LONG" in message
    assert "0.256000" in message
    assert "SL ✅ · TP ✅" in message
    assert "confidence 68%" in message
    assert "184736251" in message


def test_close_notification_includes_realized_pnl_fees_and_reason():
    message = format_trading_notification(TradingNotification(
        kind="position_closed", symbol="ETH/USDT", side="sell",
        amount=Decimal("0.256"), price=Decimal("1940"), order_id="184736252",
        realized_pnl=Decimal("2.31"), commissions=Decimal("0.49"),
        exit_reason="take_profit",
    ))

    assert "Фактический PnL" in message
    assert "$+2.31" in message
    assert "$0.49" in message
    assert "take_profit" in message
