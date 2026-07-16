from src.utils.artifact_paths import get_oos_path


def test_get_oos_path_basic():
    assert get_oos_path("models/saved_models/lgbm_BTCUSDT_1h.pkl") == \
        "models/saved_models/lgbm_BTCUSDT_1h_oos.parquet"


def test_get_oos_path_no_extension():
    assert get_oos_path("models/saved_models/lgbm_BTCUSDT_1h") == \
        "models/saved_models/lgbm_BTCUSDT_1h_oos.parquet"


def test_get_oos_path_preserves_directory():
    path = get_oos_path("/tmp/staging/lgbm_ETHUSDT_1h.pkl")
    assert path == "/tmp/staging/lgbm_ETHUSDT_1h_oos.parquet"