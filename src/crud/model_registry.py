from datetime import datetime, timedelta, timezone

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import ModelDeployment


VALID_STATUSES = {"candidate", "challenger", "champion", "retired"}


class ModelRegistryRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def register(self, **values) -> ModelDeployment:
        status = values.pop("status", "candidate")
        if status not in VALID_STATUSES:
            raise ValueError(f"Invalid model status: {status}")
        model = ModelDeployment(**values, status=status)
        self.session.add(model)
        await self.session.commit()
        await self.session.refresh(model)
        return model

    async def list(self, symbol: str, timeframe: str, target: str, status: str | None = None) -> list[ModelDeployment]:
        stmt = select(ModelDeployment).where(
            ModelDeployment.symbol == symbol,
            ModelDeployment.timeframe == timeframe,
            ModelDeployment.target == target,
        ).order_by(ModelDeployment.created_at.desc())
        if status:
            stmt = stmt.where(ModelDeployment.status == status)
        return list((await self.session.execute(stmt)).scalars())

    async def champion(self, symbol: str, timeframe: str, target: str) -> ModelDeployment | None:
        models = await self.list(symbol, timeframe, target, "champion")
        return models[0] if models else None

    async def transition(self, model_id: str, status: str, reason: str) -> None:
        if status not in VALID_STATUSES:
            raise ValueError(f"Invalid model status: {status}")
        values = {"status": status, "reason": reason}
        if status == "challenger":
            values["shadow_started_at"] = datetime.now(timezone.utc)
        if status == "champion":
            values["promoted_at"] = datetime.now(timezone.utc)
        await self.session.execute(update(ModelDeployment).where(ModelDeployment.model_id == model_id).values(**values))
        await self.session.commit()

    async def promote(self, model_id: str, symbol: str, timeframe: str, target: str, reason: str) -> None:
        await self.session.execute(update(ModelDeployment).where(
            ModelDeployment.symbol == symbol, ModelDeployment.timeframe == timeframe,
            ModelDeployment.target == target, ModelDeployment.status == "champion",
        ).values(status="retired", reason="superseded by " + model_id))
        await self.transition(model_id, "champion", reason)

    async def evaluate_shadow(
        self, model_id: str, *, min_trades: int, min_hours: int,
        champion_metrics: dict | None = None,
    ) -> bool:
        model = (await self.session.execute(select(ModelDeployment).where(
            ModelDeployment.model_id == model_id
        ))).scalar_one()
        live = model.live_metrics or {}
        shadow_since = model.shadow_started_at
        old_enough = shadow_since is not None and datetime.now(timezone.utc) - shadow_since.replace(
            tzinfo=shadow_since.tzinfo or timezone.utc
        ) >= timedelta(hours=min_hours)
        enough_trades = live.get("total_trades", 0) >= min_trades
        challenger_low = live.get("sharpe_ci_low", float("-inf"))
        champion_high = (champion_metrics or {}).get("sharpe_ci_high", float("-inf"))
        passed = old_enough and enough_trades and challenger_low > champion_high
        if not passed and old_enough and enough_trades:
            await self.transition(model_id, "retired", "shadow gate rejected: confidence intervals overlap")
        return passed
