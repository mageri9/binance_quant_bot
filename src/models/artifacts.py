"""Strict contract for deployable model artifacts."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

MODEL_ARTIFACT_SCHEMA_VERSION = 1
SUPPORTED_MODEL_TYPES = {"economic_return_regression"}


class ModelArtifactError(RuntimeError):
    pass


def feature_hash(features: list[str]) -> str:
    return hashlib.sha256(",".join(sorted(features)).encode("utf-8")).hexdigest()[:12]


def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as artifact_file:
        for chunk in iter(lambda: artifact_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ArtifactMetadata:
    model_id: str
    dataset_version: str
    schema_version: int
    features: list[str]
    features_hash: str
    model_type: str


def validate_artifact(artifact: Any, *, expected_model_id: str | None = None,
                      expected_features: list[str] | None = None) -> ArtifactMetadata:
    if not isinstance(artifact, dict):
        raise ModelArtifactError("artifact must be a dictionary")
    required = ("model", "model_id", "dataset_version", "schema_version", "features", "features_hash", "model_type")
    missing = [key for key in required if key not in artifact]
    if missing:
        raise ModelArtifactError(f"artifact metadata is missing: {', '.join(missing)}")
    if artifact["schema_version"] != MODEL_ARTIFACT_SCHEMA_VERSION:
        raise ModelArtifactError(f"unsupported schema_version={artifact['schema_version']}; expected {MODEL_ARTIFACT_SCHEMA_VERSION}")
    features = artifact["features"]
    if not isinstance(features, list) or not all(isinstance(item, str) for item in features):
        raise ModelArtifactError("artifact features must be a list of strings")
    actual_hash = feature_hash(features)
    if artifact["features_hash"] != actual_hash:
        raise ModelArtifactError("artifact features_hash does not match its feature schema")
    if expected_features is not None and features != expected_features:
        raise ModelArtifactError("artifact feature schema differs from the registry")
    if artifact["model_type"] not in SUPPORTED_MODEL_TYPES:
        raise ModelArtifactError(f"unsupported model_type={artifact['model_type']!r}")
    if expected_model_id is not None and artifact["model_id"] != expected_model_id:
        raise ModelArtifactError("artifact model_id differs from the registry champion")
    return ArtifactMetadata(artifact["model_id"], artifact["dataset_version"], artifact["schema_version"], features, actual_hash, artifact["model_type"])
