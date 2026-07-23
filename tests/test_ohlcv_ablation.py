import numpy as np
import pandas as pd

from src.features.engineering import DEFAULT_FEATURE_SCHEMA, add_features
from src.models.ablation import ohlcv_ablation_feature_sets, run_ohlcv_ablation


def test_ohlcv_ablation_compares_all_feature_families():
    rng = np.random.default_rng(7)
    returns = rng.normal(0, 0.003, 160)
    close = 100 * np.exp(np.cumsum(returns))
    df = pd.DataFrame({
        "open": close * (1 - 0.001), "high": close * 1.003,
        "low": close * 0.997, "close": close,
        "volume": rng.lognormal(8, 0.3, len(close)),
    })
    featured = add_features(df)
    featured["target_binary"] = (np.arange(len(featured)) % 2).astype(float)

    report = run_ohlcv_ablation(featured, "target_binary", train_size=60, test_size=20)

    assert len(report["results"]) == len(ohlcv_ablation_feature_sets())
    assert report["results"][0]["features"] == DEFAULT_FEATURE_SCHEMA[:11]
    assert report["results"][0]["f1_delta_vs_baseline"] is None
    assert all(result["oos_samples"] > 0 for result in report["results"])
