import os
import urllib.request

import pandas as pd
import pytest
from perpetual import PerpetualBooster

RESOURCES_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "resources")

DOWNLOAD_URLS = {
    "tabarena_x.csv": "https://github.com/user-attachments/files/25580857/X.csv",
    "tabarena_y.csv": "https://github.com/user-attachments/files/25580858/y.csv",
    "tabarena_cat_x.csv": "https://github.com/user-attachments/files/25600227/X.csv",
    "tabarena_cat_y.csv": "https://github.com/user-attachments/files/25600228/y.csv",
}


def _get_csv(filename):
    """Try to read a CSV from resources dir, then download, or skip the test."""
    local_path = os.path.join(RESOURCES_DIR, filename)
    if os.path.isfile(local_path):
        return pd.read_csv(local_path)

    # Try downloading to the resources folder
    url = DOWNLOAD_URLS.get(filename)
    if url is None:
        pytest.skip(f"No download URL configured for {filename}")
    try:
        os.makedirs(RESOURCES_DIR, exist_ok=True)
        urllib.request.urlretrieve(url, local_path)
    except Exception as exc:
        pytest.skip(f"Could not download {filename}: {exc}")

    return pd.read_csv(local_path)


def test_tabarena():
    X = _get_csv("tabarena_x.csv")
    y = _get_csv("tabarena_y.csv")

    model = PerpetualBooster(budget=1.0, categorical_features=["state", "area_code"])

    model.fit(X, y)


def test_tabarena_categorical():
    X = _get_csv("tabarena_cat_x.csv")
    y = _get_csv("tabarena_cat_y.csv")

    categorical_features = [
        "RESOURCE",
        "MGR_ID",
        "ROLE_ROLLUP_1",
        "ROLE_ROLLUP_2",
        "ROLE_DEPTNAME",
        "ROLE_TITLE",
        "ROLE_FAMILY_DESC",
        "ROLE_FAMILY",
    ]

    model = PerpetualBooster(categorical_features=categorical_features)
    model.fit(X, y)

    model = PerpetualBooster(categorical_features=categorical_features, max_cat=3000)
    model.fit(X, y)
