import pickle

import pytest

from src.crud.model_registry import ModelRegistryRepository
from src.models.artifacts import MODEL_ARTIFACT_SCHEMA_VERSION, feature_hash
from src.models.artifacts import ModelArtifactError
from src.models.predictor import Predictor
from src.models.resolver import ModelResolutionError, ModelResolver


def _artifact(model_id="champion-1", *, schema_version=MODEL_ARTIFACT_SCHEMA_VERSION,
              features=None, features_hash_value=None):
    features = features or ["rsi", "macd"]
    return {
        "model": object(), "model_id": model_id, "dataset_version": "dataset-v1",
        "schema_version": schema_version, "features": features,
        "features_hash": features_hash_value or feature_hash(features),
        "model_type": "economic_return_regression", "target_col": "expected_return",
    }


async def _champion(session, path, *, model_id="champion-1", features=None):
    return await ModelRegistryRepository(session).register(
        model_id=model_id, symbol="BTC/USDT", timeframe="1h", target="expected_return",
        status="champion", artifact_uri=str(path), feature_schema=features or ["rsi", "macd"],
        parameters={"model_type": "economic_return_regression"},
    )


@pytest.mark.asyncio
async def test_resolver_loads_registry_champion(temp_db_session, tmp_path):
    path = tmp_path / "champion.pkl"
    with path.open("wb") as artifact_file:
        pickle.dump(_artifact(), artifact_file)
    await _champion(temp_db_session, path)

    resolved = await ModelResolver(temp_db_session).resolve("BTC/USDT", "1h", "expected_return")

    assert resolved.metadata.model_id == "champion-1"
    assert resolved.deployment.status == "champion"
    assert resolved.artifact_path == str(path)
    assert len(resolved.artifact_hash) == 64


@pytest.mark.asyncio
async def test_resolver_rejects_missing_champion(temp_db_session):
    with pytest.raises(ModelResolutionError, match="no registry champion"):
        await ModelResolver(temp_db_session).resolve("BTC/USDT", "1h", "expected_return")


@pytest.mark.asyncio
async def test_resolver_rejects_missing_champion_file(temp_db_session, tmp_path):
    await _champion(temp_db_session, tmp_path / "missing.pkl")
    with pytest.raises(ModelResolutionError, match="artifact is missing"):
        await ModelResolver(temp_db_session).resolve("BTC/USDT", "1h", "expected_return")


@pytest.mark.asyncio
async def test_resolver_rejects_incompatible_schema_version(temp_db_session, tmp_path):
    path = tmp_path / "bad-schema.pkl"
    with path.open("wb") as artifact_file:
        pickle.dump(_artifact(schema_version=999), artifact_file)
    await _champion(temp_db_session, path)
    with pytest.raises(ModelResolutionError, match="schema_version"):
        await ModelResolver(temp_db_session).resolve("BTC/USDT", "1h", "expected_return")


@pytest.mark.asyncio
async def test_resolver_rejects_incompatible_feature_hash(temp_db_session, tmp_path):
    path = tmp_path / "bad-hash.pkl"
    with path.open("wb") as artifact_file:
        pickle.dump(_artifact(features_hash_value="wrong"), artifact_file)
    await _champion(temp_db_session, path)
    with pytest.raises(ModelResolutionError, match="features_hash"):
        await ModelResolver(temp_db_session).resolve("BTC/USDT", "1h", "expected_return")


@pytest.mark.asyncio
async def test_resolver_rejects_registry_model_id_not_in_artifact(temp_db_session, tmp_path):
    path = tmp_path / "other-model.pkl"
    with path.open("wb") as artifact_file:
        pickle.dump(_artifact(model_id="artifact-b"), artifact_file)
    await _champion(temp_db_session, path, model_id="registry-a")
    with pytest.raises(ModelResolutionError, match="model_id differs"):
        await ModelResolver(temp_db_session).resolve("BTC/USDT", "1h", "expected_return")


def test_predictor_rejects_legacy_artifact_without_contract(tmp_path):
    path = tmp_path / "legacy.pkl"
    with path.open("wb") as artifact_file:
        pickle.dump({"model": object()}, artifact_file)
    with pytest.raises(ModelArtifactError, match="metadata is missing"):
        Predictor(str(path))
