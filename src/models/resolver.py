"""Resolve the sole executable model from the registry champion record."""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from src.crud.model_registry import ModelRegistryRepository
from src.db.models import ModelDeployment
from src.models.artifacts import ArtifactMetadata, ModelArtifactError, file_hash, validate_artifact


class ModelResolutionError(RuntimeError):
    pass


@dataclass(frozen=True)
class ResolvedModel:
    deployment: ModelDeployment
    artifact_path: str
    artifact: dict
    metadata: ArtifactMetadata
    artifact_hash: str


class ModelResolver:
    def __init__(self, session: AsyncSession):
        self.registry = ModelRegistryRepository(session)

    async def resolve(self, symbol: str, timeframe: str, target: str) -> ResolvedModel:
        deployment = await self.registry.champion(symbol, timeframe, target)
        scope = f"{symbol} {timeframe} target={target}"
        if deployment is None:
            raise ModelResolutionError(f"no registry champion for {scope}")
        if not deployment.artifact_uri:
            raise ModelResolutionError(f"registry champion {deployment.model_id} has no artifact_uri")
        if not deployment.feature_schema:
            raise ModelResolutionError(f"registry champion {deployment.model_id} has no feature_schema")
        path = Path(deployment.artifact_uri)
        if not path.is_file():
            raise ModelResolutionError(f"registry champion artifact is missing: {path}")
        try:
            with path.open("rb") as artifact_file:
                artifact = pickle.load(artifact_file)
            metadata = validate_artifact(artifact, expected_model_id=deployment.model_id, expected_features=deployment.feature_schema)
            registry_type = (deployment.parameters or {}).get("model_type")
            if registry_type != metadata.model_type:
                raise ModelArtifactError(
                    f"registry model_type={registry_type!r} differs from artifact model_type={metadata.model_type!r}"
                )
        except (OSError, pickle.UnpicklingError, EOFError, ModelArtifactError) as exc:
            raise ModelResolutionError(f"invalid registry champion {deployment.model_id}: {exc}") from exc
        return ResolvedModel(deployment, str(path), artifact, metadata, file_hash(path))
