import numpy as np
import pandas as pd
from perpetual import PerpetualBooster


def test_multi_output():
    X = pd.read_csv("../resources/cover_types_train.csv", index_col=False)
    X = X.sample(n=10000, random_state=0)
    y = np.array(X.pop("Cover_Type"))
    X_test = pd.read_csv("../resources/cover_types_test.csv", index_col=False)
    y_test = np.array(X_test.pop("Cover_Type"))
    model = PerpetualBooster()
    model.fit(X, y, iteration_limit=40)
    pred_test = model.predict(X_test)
    proba_test = model.predict_proba(X_test)
    log_odds_test = model.predict_log_proba(X_test)
    assert not np.isnan(pred_test).any()
    assert not np.isnan(proba_test).any()
    assert not np.isnan(log_odds_test).any()
    assert np.allclose(np.sum(proba_test, axis=1), np.ones(proba_test.shape[0]))
    assert np.allclose(proba_test.shape, (len(X_test), len(np.unique(y_test))))
    assert set(y_test) == set(pred_test)
