import json
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from src.db.models import Experiment


class ExperimentRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def log_experiment(
        self,
        model_name: str,
        dataset_version: str,
        parameters: dict,
        metrics: dict,
        git_sha: str
    ) -> Experiment:
        """
        Сохраняет результаты обучения модели в базу данных.
        """
        experiment = Experiment(
            model_name=model_name,
            dataset_version=dataset_version,
            parameters=json.dumps(parameters, ensure_ascii=False),
            metrics=json.dumps(metrics, ensure_ascii=False),
            git_sha=git_sha
        )
        self.session.add(experiment)
        await self.session.commit()
        await self.session.refresh(experiment)
        return experiment

    async def list_experiments(self, model_name: str | None = None, limit: int = 100) -> list[Experiment]:
        stmt = select(Experiment).order_by(Experiment.created_at.desc()).limit(limit)
        if model_name is not None:
            stmt = stmt.where(Experiment.model_name == model_name)
        result = await self.session.execute(stmt)
        return list(result.scalars())

    async def get_metrics(self, experiment_id: int) -> dict | None:
        experiment = await self.session.get(Experiment, experiment_id)
        return json.loads(experiment.metrics) if experiment else None
