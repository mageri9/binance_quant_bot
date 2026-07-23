import json

import pytest

from src.crud.experiment import ExperimentRepository


@pytest.mark.asyncio
async def test_experiment_repository_stores_long_json_parameters(temp_db_session):
    parameters = {
        "study_name": "lgbm_" + "x" * 600,
        "search_space": {"num_leaves": [3, 128]},
    }
    repository = ExperimentRepository(temp_db_session)

    experiment = await repository.log_experiment(
        model_name="LightGBM_Hyperparameter_Tuning",
        dataset_version="test-long-json",
        parameters=parameters,
        metrics={"best_cv_sharpe_score": 0.1},
        git_sha="unknown",
    )

    assert json.loads(experiment.parameters) == parameters
