import numpy as np
import pandas as pd

from src.models.regimes import REGIME_NAMES, SoftRegimeEnsemble, soft_regime_weights


def test_soft_regime_weights_are_normalized_and_gradual():
    df = pd.DataFrame({"rsi": [45.0, 50.0, 55.0]})

    weights = soft_regime_weights(df)

    assert list(weights.columns) == list(REGIME_NAMES)
    np.testing.assert_allclose(weights.sum(axis=1), 1.0)
    assert weights.loc[0, "bear"] > weights.loc[0, "bull"]
    assert weights.loc[2, "bull"] > weights.loc[2, "bear"]
    assert weights.loc[1, "range"] > weights.loc[1, "bear"]


def test_soft_regime_weights_use_no_future_rows():
    base = pd.DataFrame({"close": np.linspace(100.0, 130.0, 40)})
    changed = base.copy()
    changed.loc[30:, "close"] *= 4

    original = soft_regime_weights(base)
    recomputed = soft_regime_weights(changed)

    pd.testing.assert_frame_equal(original.iloc[:30], recomputed.iloc[:30])


class _ConstantModel:
    def __init__(self, probabilities):
        self.classes_ = np.array([0, 1])
        self.probabilities = np.asarray(probabilities)

    def predict_proba(self, X):
        return np.tile(self.probabilities, (len(X), 1))


def test_ensemble_blends_specialists_with_current_soft_memberships():
    ensemble = SoftRegimeEnsemble(
        models=[_ConstantModel([0.9, 0.1]), _ConstantModel([0.5, 0.5]), _ConstantModel([0.1, 0.9])],
        classes_=np.array([0, 1]),
    )
    probabilities = ensemble.predict_proba(pd.DataFrame({"rsi": [30.0, 70.0]}))

    assert probabilities[0, 0] > probabilities[0, 1]
    assert probabilities[1, 1] > probabilities[1, 0]
