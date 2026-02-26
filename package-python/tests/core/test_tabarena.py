import os

os.environ["RUST_BACKTRACE"] = "full"
os.environ["PYTHONFAULTHANDLER"] = "1"
os.environ["RUST_LIB_BACKTRACE"] = "1"

import pandas as pd  # noqa: I001
from perpetual import PerpetualBooster

RESOURCES_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "resources")


def test_tabarena():
    X = pd.read_csv(os.path.join(RESOURCES_DIR, "tabarena_x.csv"))
    y = pd.read_csv(os.path.join(RESOURCES_DIR, "tabarena_y.csv"))

    model = PerpetualBooster(budget=1.0, categorical_features=["state", "area_code"])

    model.fit(X, y)
