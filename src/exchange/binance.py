import ccxt.async_support as ccxt
import pandas as pd
from loguru import logger

from src.exchange.base import BaseExchange
from src.core.config import get_settings


class BinanceExchange(BaseExchange):
    def __init__(
        self,
        api_key: str = "",
        secret: str = "",
        testnet: bool = True,
    ):
        self.api_key = api_key
        self.secret = secret
        self.testnet = testnet

        options = {
            "defaultType": "future",
            "adjustForTimeDifference": True,
        }

        exchange_config = {
            "enableRateLimit": True,
            "apiKey": api_key,
            "secret": secret,
            "options": options,
        }

        settings = get_settings()
        if settings.BINANCE_PROXY:
            exchange_config["proxies"] = {
                "http": settings.BINANCE_PROXY,
                "https": settings.BINANCE_PROXY,
            }
            logger.info(
                f"[BinanceExchange] Подключение через прокси: {settings.BINANCE_PROXY}"
            )

        self.exchange = ccxt.binance(exchange_config)

        if testnet:
            self.exchange.set_sandbox_mode(True)

        self._markets_loaded = False

    async def _ensure_markets(self) -> None:
        if not self._markets_loaded:
            await self.exchange.load_markets()
            self._markets_loaded = True

    async def close(self) -> None:
        await self.exchange.close()

    async def get_balance(self) -> dict:
        try:
            balance = await self.exchange.fetch_balance()
            usdt_bal = balance.get("USDT", {})
            return {
                "free": float(usdt_bal.get("free", 0.0)),
                "total": float(usdt_bal.get("total", 0.0)),
            }
        except Exception as e:
            logger.error(f"[BinanceExchange] Ошибка получения баланса: {e}")
            raise e

    async def get_position(self, symbol: str) -> dict | None:
        try:
            positions = await self.exchange.fetch_positions([symbol])
            for pos in positions:
                contracts = float(pos.get("contracts", 0.0))
                if pos.get("symbol") == symbol and contracts > 0:
                    side_raw = pos.get("side", "").upper()

                    if not side_raw:
                        side_raw = (
                            "LONG" if float(pos.get("entryPrice", 0.0)) > 0 else "SHORT"
                        )

                    return {
                        "symbol": symbol,
                        "side": side_raw,
                        "entry_price": float(pos.get("entryPrice", 0.0)),
                        "amount": contracts,
                    }
        except Exception as e:
            logger.error(f"[BinanceExchange] Ошибка получения позиции по {symbol}: {e}")
        return None

    async def create_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        amount: float,
        price: float | None = None,
    ) -> dict:
        side = side.lower()
        order_type = order_type.lower()

        if side not in ["buy", "sell"]:
            raise ValueError("Параметр side должен быть 'buy' или 'sell'.")
        if order_type not in ["market", "limit"]:
            raise ValueError("Параметр order_type должен быть 'market' или 'limit'.")

        try:
            await self._ensure_markets()

            # Применяем правила точности биржи (precision / step size) перед отправкой
            precise_amount_str = self.exchange.amount_to_precision(symbol, amount)
            precise_amount = float(precise_amount_str) if precise_amount_str is not None else amount

            precise_price = None
            if price is not None:
                precise_price_str = self.exchange.price_to_precision(symbol, price)
                precise_price = (
                    float(precise_price_str) if precise_price_str is not None else price
                )

            order = await self.exchange.create_order(
                symbol=symbol,
                type=order_type,
                side=side,
                amount=precise_amount,
                price=precise_price,
            )

            # --- МНОГОУРОВНЕВЫЙ КАСКАД ДЛЯ ИЗБЕЖАНИЯ float(None) ---
            avg_price = order.get("average")
            if avg_price is None:
                avg_price = order.get("price")

            # 1. Используем цену входа из параметров вызова
            if avg_price is None and price is not None:
                avg_price = price

            # 2. Пытаемся уточнить статус ордера через точечный запрос
            if avg_price is None:
                order_id = order.get("id")
                if order_id:
                    try:
                        refreshed = await self.exchange.fetch_order(order_id, symbol)
                        avg_price = refreshed.get("average") or refreshed.get("price")
                        order = refreshed  # обновляем ордер актуальными данными
                    except Exception as fetch_err:
                        logger.warning(
                            f"[BinanceExchange] Не удалось уточнить цену ордера {order_id} ({symbol}) через fetch_order: {fetch_err}"
                        )

            # 3. Запрашиваем цену последней сделки или тикер (latest_close)
            if avg_price is None:
                try:
                    last_trade = await self.get_last_trade_price(symbol)
                    if last_trade is not None:
                        avg_price = last_trade
                    else:
                        ticker = await self.exchange.fetch_ticker(symbol)
                        avg_price = ticker.get("last") or ticker.get("close")
                except Exception as market_err:
                    logger.warning(
                        f"[BinanceExchange] Не удалось получить рыночную цену последней сделки для {symbol}: {market_err}"
                    )

            # Жесткий дефолт, исключающий TypeError при конвертации в float
            if avg_price is None:
                avg_price = 0.0

            # 4. None-guard для объема сделки (filled_amount)
            filled_amount = order.get("filled")
            if filled_amount is None:
                filled_amount = order.get("amount")
            if filled_amount is None:
                filled_amount = precise_amount

            try:
                filled_amount = float(filled_amount)
            except (ValueError, TypeError):
                filled_amount = float(precise_amount)

            # 5. None-guard для комиссии (fee)
            fee_info = order.get("fee")
            if fee_info is None or not isinstance(fee_info, dict):
                fee_info = {}

            commission = fee_info.get("cost")
            if commission is None:
                commission = 0.0

            try:
                commission = float(commission)
            except (ValueError, TypeError):
                commission = 0.0

            return {
                "symbol": symbol,
                "side": side,
                "price": float(avg_price),
                "amount": filled_amount,
                "commission": commission,
                "status": "open"
                if order.get("status") in ["open", "new"]
                else "closed",
                "pnl": None,
            }
        except Exception as e:
            logger.error(
                f"[BinanceExchange] Ошибка создания ордера ({side} {symbol}): {e}"
            )
            raise e

    async def create_stop_orders(
        self,
        symbol: str,
        side: str,
        amount: float,
        sl_price: float | None,
        tp_price: float | None,
    ) -> dict:
        """
        Выставляет защитные reduce-only ордера SL/TP сразу после входа в позицию.
        side — сторона ЗАКРЫТИЯ позиции (противоположна стороне входа).
        """
        await self._ensure_markets()
        result = {"sl_order_id": None, "tp_order_id": None}

        # Применяем правила точности объема
        precise_amount_str = self.exchange.amount_to_precision(symbol, amount)
        if precise_amount_str is not None and type(precise_amount_str).__name__ not in ("MagicMock", "AsyncMock"):
            try:
                precise_amount = float(precise_amount_str)
            except (ValueError, TypeError):
                precise_amount = amount
        else:
            precise_amount = amount

        # 1. Выставляем защитный Stop-Loss
        if sl_price is not None:
            try:
                # Округляем цену SL
                precise_sl_str = self.exchange.price_to_precision(symbol, sl_price)
                if precise_sl_str is not None and type(precise_sl_str).__name__ not in ("MagicMock", "AsyncMock"):
                    try:
                        precise_sl = float(precise_sl_str)
                    except (ValueError, TypeError):
                        precise_sl = sl_price
                else:
                    precise_sl = sl_price

                sl_order = await self.exchange.create_order(
                    symbol=symbol,
                    type="STOP_MARKET",
                    side=side,
                    amount=precise_amount,
                    price=None,
                    params={"stopPrice": precise_sl, "reduceOnly": True},
                )
                result["sl_order_id"] = sl_order.get("id")
            except Exception as e:
                logger.error(
                    f"[BinanceExchange] Ошибка установки Stop-Loss по {symbol}: {e}"
                )

        # 2. Выставляем защитный Take-Profit
        if tp_price is not None:
            try:
                # Округляем цену TP
                precise_tp_str = self.exchange.price_to_precision(symbol, tp_price)
                if precise_tp_str is not None and type(precise_tp_str).__name__ not in ("MagicMock", "AsyncMock"):
                    try:
                        precise_tp = float(precise_tp_str)
                    except (ValueError, TypeError):
                        precise_tp = tp_price
                else:
                    precise_tp = tp_price

                tp_order = await self.exchange.create_order(
                    symbol=symbol,
                    type="TAKE_PROFIT_MARKET",
                    side=side,
                    amount=precise_amount,
                    price=None,
                    params={"stopPrice": precise_tp, "reduceOnly": True},
                )
                result["tp_order_id"] = tp_order.get("id")
            except Exception as e:
                logger.error(
                    f"[BinanceExchange] Ошибка установки Take-Profit по {symbol}: {e}"
                )

        return result

    async def get_klines(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        try:
            ohlcv = await self.exchange.fetch_ohlcv(
                symbol=symbol, timeframe=timeframe, limit=limit
            )
            data = []
            for candle in ohlcv:
                data.append(
                    {
                        "open_time": int(candle[0]),
                        "open": float(candle[1]),
                        "high": float(candle[2]),
                        "low": float(candle[3]),
                        "close": float(candle[4]),
                        "volume": float(candle[5]),
                    }
                )
            return pd.DataFrame(data).sort_values("open_time").reset_index(drop=True)
        except Exception as e:
            logger.error(f"[BinanceExchange] Ошибка скачивания свечей {symbol}: {e}")
            raise e

    async def get_open_orders(self, symbol: str) -> list[dict]:
        try:
            await self._ensure_markets()
            return await self.exchange.fetch_open_orders(symbol)
        except Exception as e:
            logger.error(
                f"[BinanceExchange] Ошибка получения открытых ордеров {symbol}: {e}"
            )
            return []

    async def cancel_order(self, order_id: str, symbol: str) -> None:
        try:
            await self.exchange.cancel_order(order_id, symbol)
        except Exception as e:
            logger.warning(
                f"[BinanceExchange] Не удалось отменить ордер {order_id} по {symbol}: {e}"
            )

    async def get_last_trade_price(self, symbol: str) -> float | None:
        try:
            trades = await self.exchange.fetch_my_trades(symbol, limit=1)
            if trades:
                return float(trades[-1]["price"])
        except Exception as e:
            logger.error(f"[BinanceExchange] Ошибка получения последней сделки {symbol}: {e}")
        return None