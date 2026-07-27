import ccxt.async_support as ccxt
import pandas as pd
from loguru import logger
from src.exchange.base import BaseExchange

class BinanceExchange(BaseExchange):
    """Binance Futures коннектор (Testnet & Mainnet)."""
    def __init__(self, api_key: str, secret: str, testnet: bool = True, proxy: str = ""):
        config = {
            "apiKey": api_key,
            "secret": secret,
            "enableRateLimit": True,
            "options": {"defaultType": "future", "adjustForTimeDifference": True},
        }
        if proxy:
            config["proxies"] = {"http": proxy, "https": proxy}
        self.exchange = ccxt.binance(config)
        if testnet:
            self.exchange.set_sandbox_mode(True)

    async def _ensure_markets(self):
        """Загружает рынки Binance, если они еще не загружены в память."""
        if not self.exchange.markets:
            await self.exchange.load_markets()

    async def close(self):
        await self.exchange.close()

    async def get_balance(self) -> dict:
        bal = await self.exchange.fetch_balance()
        usdt = bal.get("USDT", {})
        return {"free": float(usdt.get("free", 0.0)), "total": float(usdt.get("total", 0.0))}

    async def get_position(self, symbol: str) -> dict | None:
        positions = await self.exchange.fetch_positions([symbol])
        clean_target = symbol.replace("/", "").replace(":", "").upper()
        for pos in positions:
            pos_sym = str(pos.get("symbol", "")).replace("/", "").replace(":", "").upper()
            contracts = abs(float(pos.get("contracts") or pos.get("info", {}).get("positionAmt") or 0.0))
            if pos_sym == clean_target and contracts > 0:
                return {
                    "symbol": symbol,
                    "side": pos.get("side", "").upper(),
                    "entry_price": float(pos.get("entryPrice", 0.0)),
                    "amount": contracts,
                }
        return None

    async def create_order(
        self, symbol: str, side: str, order_type: str, amount: float, price: float | None = None, reduce_only: bool = False
    ) -> dict:
        await self._ensure_markets()
        formatted_amount = float(self.exchange.amount_to_precision(symbol, amount))
        formatted_price = float(self.exchange.price_to_precision(symbol, price)) if price else None

        params = {}
        if reduce_only:
            params["reduceOnly"] = True
        order = await self.exchange.create_order(
            symbol=symbol, type=order_type.lower(), side=side.lower(), amount=formatted_amount, price=formatted_price, params=params
        )
        return {
            "order_id": str(order.get("id")),
            "symbol": symbol,
            "side": side,
            "amount": float(order.get("amount", formatted_amount)),
            "price": float(order.get("average") or order.get("price") or 0.0),
            "status": str(order.get("status")).upper()
        }

    async def create_stop_orders(
        self, symbol: str, side: str, amount: float, sl_price: float, tp_price: float
    ) -> dict:
        """Установка условных стоп-ордеров через Algo Order API Binance с форматированием точности."""
        await self._ensure_markets()
        formatted_amount = self.exchange.amount_to_precision(symbol, amount)

        async def _place(order_type: str, trigger_price: float):
            market_sym = symbol.replace("/", "")
            formatted_price = self.exchange.price_to_precision(symbol, trigger_price)
            params = {
                "symbol": market_sym,
                "side": side.upper(),
                "type": order_type,
                "algoType": "CONDITIONAL",
                "triggerPrice": str(formatted_price),
                "quantity": str(formatted_amount),
                "reduceOnly": "true",
            }
            if hasattr(self.exchange, "fapiPrivatePostAlgoOrder"):
                return await self.exchange.fapiPrivatePostAlgoOrder(params)
            return await self.exchange.request("algoOrder", "fapiPrivate", "POST", params)

        sl_resp = await _place("STOP_MARKET", sl_price) if sl_price else None
        tp_resp = await _place("TAKE_PROFIT_MARKET", tp_price) if tp_price else None
        return {"sl_order": sl_resp, "tp_order": tp_resp}

    async def get_klines(self, symbol: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
        ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        data = [{
            "open_time": int(c[0]), "open": float(c[1]), "high": float(c[2]),
            "low": float(c[3]), "close": float(c[4]), "volume": float(c[5])
        } for c in ohlcv]
        return pd.DataFrame(data)
