import asyncio

import aiohttp
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
            "timeout": 15000,
        }

        settings = get_settings()
        if settings.BINANCE_PROXY:
            exchange_config["proxies"] = {
                "http": settings.BINANCE_PROXY,
                "https": settings.BINANCE_PROXY,
            }
            logger.debug(
                f"[BinanceExchange] Подключение через прокси: {settings.BINANCE_PROXY}"
            )

        self.exchange = ccxt.binance(exchange_config)

        if testnet:
            self.exchange.set_sandbox_mode(True)

        self._markets_loaded = False
        self._user_stream_session: aiohttp.ClientSession | None = None

    async def _ensure_markets(self) -> None:
        if not self._markets_loaded:
            await self.exchange.load_markets()
            self._markets_loaded = True

    async def close(self) -> None:
        if self._user_stream_session is not None:
            await self._user_stream_session.close()
            self._user_stream_session = None
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
                        "mark_price": _optional_float(pos.get("markPrice")),
                        "unrealized_pnl": _optional_float(
                            pos.get("unrealizedPnl")
                            if pos.get("unrealizedPnl") is not None
                            else pos.get("unrealizedProfit")
                        ),
                        "leverage": _optional_float(pos.get("leverage")),
                        "timestamp": pos.get("timestamp")
                        or (pos.get("info") or {}).get("updateTime"),
                        "raw": pos,
                    }
        except Exception as e:
            logger.error(f"[BinanceExchange] Ошибка получения позиции по {symbol}: {e}")
            # A transport failure is not evidence of a flat account. Callers
            # must enter SAFE_MODE instead of closing their local projection.
            raise
        return None

    async def create_listen_key(self) -> str:
        response = await self.exchange.request("listenKey", "fapiPrivate", "POST", {})
        listen_key = (response or {}).get("listenKey")
        if not listen_key:
            raise RuntimeError(f"Binance did not return listenKey: {response!r}")
        return str(listen_key)

    async def keepalive_listen_key(self, listen_key: str) -> None:
        await self.exchange.request(
            "listenKey", "fapiPrivate", "PUT", {"listenKey": listen_key}
        )

    async def close_listen_key(self, listen_key: str) -> None:
        try:
            await self.exchange.request(
                "listenKey", "fapiPrivate", "DELETE", {"listenKey": listen_key}
            )
        except Exception as exc:
            logger.warning(f"[BinanceExchange] Не удалось закрыть listenKey: {exc}")

    async def user_data_stream(self):
        """Yield futures User Data Stream events and reconnect with a fresh key."""
        base_url = (
            "wss://stream.binancefuture.com/ws"
            if self.testnet
            else "wss://fstream.binance.com/ws"
        )
        if self._user_stream_session is None or self._user_stream_session.closed:
            self._user_stream_session = aiohttp.ClientSession()

        while True:
            listen_key = None
            try:
                listen_key = await self.create_listen_key()
                yield {"e": "_STREAM_RECONNECTED", "listen_key": listen_key}
                ws_url = f"{base_url}/{listen_key}"
                async with self._user_stream_session.ws_connect(
                    ws_url, heartbeat=20, receive_timeout=65
                ) as socket:
                    while True:
                        try:
                            message = await asyncio.wait_for(socket.receive(), timeout=45)
                        except asyncio.TimeoutError:
                            await self.keepalive_listen_key(listen_key)
                            continue
                        if message.type == aiohttp.WSMsgType.TEXT:
                            payload = message.json()
                            if isinstance(payload, dict):
                                yield payload
                        elif message.type in {
                            aiohttp.WSMsgType.CLOSED,
                            aiohttp.WSMsgType.CLOSING,
                            aiohttp.WSMsgType.ERROR,
                        }:
                            raise ConnectionError("Binance User Data Stream disconnected")
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error(f"[BinanceExchange] User Data Stream reconnect: {exc}")
                await asyncio.sleep(2)
            finally:
                if listen_key:
                    await self.close_listen_key(listen_key)

    async def create_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        amount: float,
        price: float | None = None,
        client_order_id: str | None = None,
        reduce_only: bool = False,
    ) -> dict:
        side = side.lower()
        order_type = order_type.lower()

        if side not in ["buy", "sell"]:
            raise ValueError("Параметр side должен быть 'buy' или 'sell'.")
        if order_type not in ["market", "limit"]:
            raise ValueError("Параметр order_type должен быть 'market' или 'limit'.")

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

        if client_order_id:
            try:
                existing = await self.get_order_by_client_id(symbol, client_order_id)
                if existing is not None:
                    existing["recovered"] = True
                    return existing
            except Exception as lookup_err:
                logger.warning(
                    f"[BinanceExchange] Предварительный поиск {client_order_id} не удался: {lookup_err}"
                )

        params = {}
        if client_order_id:
            params["newClientOrderId"] = client_order_id
        if reduce_only:
            params["reduceOnly"] = True

        try:
            order = await self.exchange.create_order(
                symbol=symbol,
                type=order_type,
                side=side,
                amount=precise_amount,
                price=precise_price,
                params=params,
            )
        except Exception as e:
            # Ответ мог потеряться после исполнения. Идемпотентный client ID
            # позволяет восстановить факт ордера вместо повторного market-order.
            if client_order_id:
                try:
                    recovered = await self.get_order_by_client_id(symbol, client_order_id)
                    if recovered is not None:
                        recovered["recovered"] = True
                        return recovered
                except Exception as recovery_err:
                    logger.error(
                        f"[BinanceExchange] Не удалось восстановить {client_order_id}: {recovery_err}"
                    )
            logger.error(f"[BinanceExchange] Ошибка создания ордера ({side} {symbol}): {e}")
            raise

        return await self._normalize_order(
            order,
            symbol=symbol,
            side=side,
            fallback_amount=precise_amount,
            fallback_price=precise_price,
            client_order_id=client_order_id,
        )

    async def get_order_by_client_id(
        self, symbol: str, client_order_id: str
    ) -> dict | None:
        await self._ensure_markets()
        params = {
            "symbol": self.exchange.market_id(symbol),
            "origClientOrderId": client_order_id,
        }
        try:
            raw = await self.exchange.request("order", "fapiPrivate", "GET", params)
        except Exception as exc:
            message = str(exc).lower()
            if "-2013" in message or "order does not exist" in message or "unknown order" in message:
                return None
            raise
        if not raw:
            return None
        return await self._normalize_order(
            raw,
            symbol=symbol,
            side=str(raw.get("side") or "").lower(),
            fallback_amount=_optional_float(raw.get("origQty")) or 0.0,
            fallback_price=_optional_float(raw.get("price")),
            client_order_id=client_order_id,
        )

    async def _normalize_order(
        self,
        order: dict,
        *,
        symbol: str,
        side: str,
        fallback_amount: float,
        fallback_price: float | None,
        client_order_id: str | None,
    ) -> dict:
        info = order.get("info") if isinstance(order.get("info"), dict) else {}
        raw_status = str(order.get("status") or info.get("status") or "unknown")
        normalized_status = {
            "closed": "filled",
            "filled": "filled",
            "open": "open",
            "new": "open",
            "partially_filled": "partially_filled",
            "canceled": "canceled",
            "cancelled": "canceled",
            "rejected": "rejected",
            "expired": "expired",
        }.get(raw_status.lower(), raw_status.lower())

        average_price = _first_float(
            order.get("average"),
            order.get("price"),
            info.get("avgPrice"),
            info.get("price"),
            fallback_price,
        )
        if not average_price:
            order_id = order.get("id") or info.get("orderId")
            if order_id and not info:
                try:
                    refreshed = await self.exchange.fetch_order(str(order_id), symbol)
                    return await self._normalize_order(
                        refreshed,
                        symbol=symbol,
                        side=side,
                        fallback_amount=fallback_amount,
                        fallback_price=fallback_price,
                        client_order_id=client_order_id,
                    )
                except Exception as fetch_err:
                    logger.warning(
                        f"[BinanceExchange] Не удалось уточнить ордер {order_id}: {fetch_err}"
                    )
        if not average_price:
            average_price = await self.get_last_trade_price(symbol)
        if not average_price:
            ticker = await self.exchange.fetch_ticker(symbol)
            average_price = _first_float(ticker.get("last"), ticker.get("close"), 0.0)

        filled_amount = _first_float(
            order.get("filled"), info.get("executedQty"), info.get("cumQty")
        )
        requested_amount = _first_float(
            order.get("amount"), info.get("origQty"), fallback_amount
        )
        if filled_amount is None:
            filled_amount = requested_amount if normalized_status == "filled" else 0.0

        fee = order.get("fee") if isinstance(order.get("fee"), dict) else {}
        fees = order.get("fees") if isinstance(order.get("fees"), list) else []
        commission = _optional_float(fee.get("cost")) or sum(
            _optional_float(item.get("cost")) or 0.0 for item in fees
        )
        commission_asset = fee.get("currency") or next(
            (item.get("currency") for item in fees if item.get("currency")), None
        )

        fills = []
        for trade in order.get("trades") or []:
            trade_fee = trade.get("fee") if isinstance(trade.get("fee"), dict) else {}
            fills.append(
                {
                    "trade_id": trade.get("id") or (trade.get("info") or {}).get("id"),
                    "order_id": trade.get("order") or order.get("id") or info.get("orderId"),
                    "client_order_id": client_order_id,
                    "symbol": symbol,
                    "side": side,
                    "price": _first_float(trade.get("price"), average_price),
                    "amount": _first_float(trade.get("amount"), trade.get("filled"), 0.0),
                    "commission": _optional_float(trade_fee.get("cost")) or 0.0,
                    "commission_asset": trade_fee.get("currency"),
                    "timestamp": trade.get("timestamp"),
                    "raw": trade,
                }
            )

        return {
            "symbol": symbol,
            "side": side.lower(),
            "order_id": str(order.get("id") or info.get("orderId"))
            if order.get("id") is not None or info.get("orderId") is not None
            else None,
            "client_order_id": order.get("clientOrderId")
            or info.get("clientOrderId")
            or info.get("origClientOrderId")
            or client_order_id,
            "price": float(average_price or 0.0),
            "average_price": float(average_price or 0.0),
            "amount": float(requested_amount or 0.0),
            "filled_amount": float(filled_amount or 0.0),
            "remaining_amount": _optional_float(order.get("remaining")),
            "commission": float(commission),
            "commission_asset": commission_asset,
            "status": normalized_status,
            "raw_status": raw_status,
            "timestamp": order.get("timestamp")
            or info.get("updateTime")
            or info.get("transactTime"),
            "fills": fills,
            "raw": order,
            "realized_pnl": _optional_float(order.get("realizedPnl") or info.get("realizedPnl")),
        }

    async def create_stop_orders(
        self,
        symbol: str,
        side: str,
        amount: float,
        sl_price: float | None,
        tp_price: float | None,
        sl_client_order_id: str | None = None,
        tp_client_order_id: str | None = None,
    ) -> dict:
        """
        Выставляет защитные reduce-only ордера SL/TP через новый Algo Order API
        (POST /fapi/v1/algoOrder). С 2025-12-09 Binance перевёл все conditional-ордера
        (STOP_MARKET/TAKE_PROFIT_MARKET) на этот эндпоинт, старый /fapi/v1/order
        их больше не принимает (код -4120).
        side — сторона ЗАКРЫТИЯ позиции (противоположна стороне входа).
        """
        await self._ensure_markets()
        result = {
            "sl_order_id": None,
            "tp_order_id": None,
            "sl_order": None,
            "tp_order": None,
        }

        precise_amount_str = self.exchange.amount_to_precision(symbol, amount)
        try:
            precise_amount = float(precise_amount_str)
        except (ValueError, TypeError):
            precise_amount = amount

        async def _place_algo_order(
            order_type: str, trigger_price: float, client_order_id: str | None
        ) -> dict | None:
            try:
                precise_trigger_str = self.exchange.price_to_precision(symbol, trigger_price)
                try:
                    precise_trigger = float(precise_trigger_str)
                except (ValueError, TypeError):
                    precise_trigger = trigger_price

                params = {
                    "algoType": "CONDITIONAL",
                    "symbol": self.exchange.market_id(symbol),
                    "side": side.upper(),
                    "type": order_type,
                    "quantity": precise_amount,
                    "triggerPrice": precise_trigger,
                    "reduceOnly": "true",
                }
                if client_order_id:
                    params["clientAlgoId"] = client_order_id
                response = await self.exchange.request(
                    "algoOrder", "fapiPrivate", "POST", params
                )
                if not response:
                    return None
                return {
                    "symbol": symbol,
                    "side": side.lower(),
                    "order_id": str(response.get("algoId")) if response.get("algoId") is not None else None,
                    "client_order_id": response.get("clientAlgoId") or client_order_id,
                    "amount": precise_amount,
                    "filled_amount": 0,
                    "price": precise_trigger,
                    "status": response.get("algoStatus") or "NEW",
                    "raw_status": response.get("algoStatus") or "NEW",
                    "timestamp": response.get("updateTime") or response.get("createTime"),
                    "raw": response,
                }
            except Exception as e:
                logger.error(
                    f"[BinanceExchange] Ошибка установки {order_type} по {symbol}: {e}"
                )
                return None

        if sl_price is not None:
            result["sl_order"] = await _place_algo_order(
                "STOP_MARKET", sl_price, sl_client_order_id
            )
            result["sl_order_id"] = (result["sl_order"] or {}).get("order_id")

        if tp_price is not None:
            result["tp_order"] = await _place_algo_order(
                "TAKE_PROFIT_MARKET", tp_price, tp_client_order_id
            )
            result["tp_order_id"] = (result["tp_order"] or {}).get("order_id")

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
            regular_raw = await self.exchange.fetch_open_orders(symbol)
            regular = [
                {
                    **o,
                    "id": str(o.get("id")),
                    "client_order_id": o.get("clientOrderId")
                    or (o.get("info") or {}).get("clientOrderId"),
                    "is_algo": False,
                }
                for o in regular_raw
            ]

            algo_response = await self.exchange.request(
                "openAlgoOrders",
                "fapiPrivate",
                "GET",
                {"symbol": self.exchange.market_id(symbol)},
            )
            raw_algo_orders = (
                algo_response.get("orders", [])
                if isinstance(algo_response, dict)
                else (algo_response or [])
            )
            algo_orders = [
                {
                    "id": str(o.get("algoId")),
                    "symbol": symbol,
                    "type": o.get("orderType") or o.get("type"),
                    "side": (o.get("side") or "").lower(),
                    "client_order_id": o.get("clientAlgoId"),
                    "is_algo": True,
                }
                for o in raw_algo_orders
            ]
            return list(regular) + algo_orders
        except Exception as e:
            logger.error(
                f"[BinanceExchange] Ошибка получения открытых ордеров {symbol}: {e}"
            )
            return []

    async def cancel_order(self, order_id: str, symbol: str, is_algo: bool = False) -> None:
        try:
            if is_algo:
                await self.exchange.request(
                    "algoOrder",
                    "fapiPrivate",
                    "DELETE",
                    {"symbol": self.exchange.market_id(symbol), "algoId": order_id},
                )
            else:
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

    async def get_recent_fills(self, symbol: str, since: int | None = None) -> list[dict]:
        """REST control snapshot used only for reconciliation, not live state."""
        await self._ensure_markets()
        trades = await self.exchange.fetch_my_trades(symbol, since=since, limit=1000)
        fills = []
        for trade in trades:
            fee = trade.get("fee") if isinstance(trade.get("fee"), dict) else {}
            info = trade.get("info") or {}
            fills.append(
                {
                    "trade_id": trade.get("id") or info.get("id"),
                    "order_id": trade.get("order") or info.get("orderId"),
                    "client_order_id": info.get("clientOrderId"),
                    "symbol": symbol,
                    "side": trade.get("side"),
                    "price": trade.get("price"),
                    "amount": trade.get("amount"),
                    "commission": fee.get("cost") or info.get("commission") or 0,
                    "commission_asset": fee.get("currency") or info.get("commissionAsset"),
                    "realized_pnl": info.get("realizedPnl"),
                    "timestamp": trade.get("timestamp") or info.get("time"),
                    "raw": trade,
                }
            )
        return fills


def _optional_float(value) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_float(*values) -> float | None:
    for value in values:
        parsed = _optional_float(value)
        if parsed is not None:
            return parsed
    return None
