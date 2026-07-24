import numpy as np
import pandas as pd

from src.models.economic import EconomicReturnRegressor


def test_economic_regressor_selects_side_by_predicted_post_cost_return():
    X = pd.DataFrame({"feature": np.arange(24, dtype=float)})
    y = pd.DataFrame({
        "long_net_return": np.linspace(-0.02, 0.03, 24),
        "short_net_return": np.linspace(0.03, -0.02, 24),
    })
    model = EconomicReturnRegressor(
        n_estimators=20, learning_rate=0.2, num_leaves=4,
        min_child_samples=1, verbosity=-1, n_jobs=1,
    )
    model.fit(X, y)

    X_test = pd.DataFrame({"feature": [0.0, 23.0]})
    long_return, short_return = model.predict_returns(X_test)

    assert short_return[0] > long_return[0]
    assert long_return[1] > short_return[1]
    assert model.signals(X_test).tolist() == [-1, 1]
