import pandas as pd
from loguru import logger
from sqlalchemy import select

from src.db import AsyncSessionFactory, Kline, PredictionLog
from src.event_bus import AsyncEventBus, CandleClosedEvent, SignalEmittedEvent
from src.strategy.features import add_features
from src.strategy.generator import SignalGenerator


class StrategyService:
    """Слушает CandleClosedEvent -> Запускает ML -> Публикует SignalEmittedEvent."""
    def __init__(self, bus: AsyncEventBus, model_dir: str = "models/saved_models"):
        self.bus = bus
        self.generator = SignalGenerator(model_dir)
        self.bus.subscribe(CandleClosedEvent, self.on_candle_closed)

    async def on_candle_closed(self, event: CandleClosedEvent):
        async with AsyncSessionFactory() as session:
            stmt = select(Kline).where(
                Kline.symbol == event.symbol,
                Kline.timeframe == event.timeframe
            ).order_by(Kline.open_time.desc()).limit(100)

            res = await session.execute(stmt)
            klines = list(reversed(res.scalars().all()))

            if len(klines) < 30:
                return

            df = pd.DataFrame([{
                "open_time": k.open_time, "open": k.open, "high": k.high,
                "low": k.low, "close": k.close, "volume": k.volume
            } for k in klines])

            df_feat = add_features(df)
            atr_val = float(df_feat.iloc[-1]["atr"])
            signal, expected_return = self.generator.generate_from_features(
                df_feat, symbol=event.symbol, timeframe=event.timeframe
            )

            # Логируем прогноз в БД
            log = PredictionLog(
                symbol=event.symbol,
                timeframe=event.timeframe,
                signal=signal,
                expected_return=expected_return,
                price=event.close
            )
            session.add(log)
            await session.commit()

            if signal != 0:
                logger.info(f"[StrategyService] Сигнал {event.symbol}: {signal} (EV={expected_return:.4f})")
                await self.bus.publish(SignalEmittedEvent(
                    symbol=event.symbol,
                    timeframe=event.timeframe,
                    signal=signal,
                    expected_return=expected_return,
                    close_price=event.close,
                    open_time=event.open_time,
                    atr=atr_val,
                ))
