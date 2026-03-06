import os
import pickle
import time
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
    "tabarena_wide_x.csv": "https://github.com/user-attachments/files/25682565/X.csv",
    "tabarena_wide_y.csv": "https://github.com/user-attachments/files/25682566/y.csv",
    "tabarena_var_x.csv": "https://github.com/user-attachments/files/25739177/X.csv",
    "tabarena_var_y.csv": "https://github.com/user-attachments/files/25739178/y.csv",
    "tabarena_load_x.csv": "https://github.com/user-attachments/files/25764663/X.csv",
    "tabarena_load_y.csv": "https://github.com/user-attachments/files/25764665/y.csv",
}


def _get_csv(filename):
    """Try to read a CSV from resources dir, then download, or skip the test."""
    print(f"Loading {filename}...")
    local_path = os.path.join(RESOURCES_DIR, filename)
    if os.path.isfile(local_path):
        res = pd.read_csv(local_path)
        print(f"Loaded {filename} from resources.")
        return res

    # Try downloading to the resources folder
    url = DOWNLOAD_URLS.get(filename)
    if url is None:
        pytest.skip(f"No download URL configured for {filename}")
    try:
        print(f"Downloading {filename} from {url}...")
        os.makedirs(RESOURCES_DIR, exist_ok=True)
        urllib.request.urlretrieve(url, local_path)
        print(f"Downloaded {filename}.")
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


def test_tabarena_wide():
    X = _get_csv("tabarena_wide_x.csv")
    y = _get_csv("tabarena_wide_y.csv")

    categorical_features = [
        "INF_ANAM",
        "STENOK_AN",
        "FK_STENOK",
        "IBS_POST",
        "IBS_NASL",
        "GB",
        "SIM_GIPERT",
        "DLIT_AG",
        "ZSN_A",
        "nr_11",
        "nr_01",
        "nr_02",
        "nr_03",
        "nr_04",
        "nr_07",
        "nr_08",
        "np_01",
        "np_04",
        "np_05",
        "np_08",
        "np_09",
        "endocr_01",
        "endocr_02",
        "endocr_03",
        "zab_leg_01",
        "zab_leg_02",
        "zab_leg_03",
        "zab_leg_04",
        "zab_leg_06",
        "O_L_POST",
        "K_SH_POST",
        "MP_TP_POST",
        "SVT_POST",
        "GT_POST",
        "FIB_G_POST",
        "ant_im",
        "lat_im",
        "inf_im",
        "post_im",
        "IM_PG_P",
        "ritm_ecg_p_01",
        "ritm_ecg_p_02",
        "ritm_ecg_p_04",
        "ritm_ecg_p_07",
        "ritm_ecg_p_08",
        "n_r_ecg_p_01",
        "n_r_ecg_p_02",
        "n_r_ecg_p_03",
        "n_r_ecg_p_04",
        "n_r_ecg_p_05",
        "n_r_ecg_p_06",
        "n_r_ecg_p_08",
        "n_r_ecg_p_09",
        "n_r_ecg_p_10",
        "n_p_ecg_p_01",
        "n_p_ecg_p_03",
        "n_p_ecg_p_04",
        "n_p_ecg_p_05",
        "n_p_ecg_p_06",
        "n_p_ecg_p_07",
        "n_p_ecg_p_08",
        "n_p_ecg_p_09",
        "n_p_ecg_p_10",
        "n_p_ecg_p_11",
        "n_p_ecg_p_12",
        "fibr_ter_01",
        "fibr_ter_02",
        "fibr_ter_03",
        "fibr_ter_05",
        "fibr_ter_06",
        "fibr_ter_07",
        "fibr_ter_08",
        "GIPO_K",
        "GIPER_NA",
        "TIME_B_S",
        "R_AB_1_n",
        "R_AB_2_n",
        "R_AB_3_n",
        "NA_KB",
        "NOT_NA_KB",
        "LID_KB",
        "NITR_S",
        "NOT_NA_1_n",
        "LID_S_n",
        "B_BLOK_S_n",
        "ANT_CA_S_n",
        "GEPAR_S_n",
        "ASP_S_n",
        "TIKL_S_n",
        "TRENT_S_n",
    ]

    model = PerpetualBooster(
        objective="LogLoss",
        budget=2.0,
        categorical_features=categorical_features,
        memory_limit=1,
        iteration_limit=3,
        timeout=60 * 15,
    )

    start_time = time.time()
    model.fit(X, y)
    end_time = time.time()

    print(f"Trees: {model.number_of_trees}")
    print(f"Fit time: {end_time - start_time}")


def test_tabarena_var():
    X = _get_csv("tabarena_var_x.csv")
    y = _get_csv("tabarena_var_y.csv")

    categorical_features = [
        "Var192",
        "Var193",
        "Var194",
        "Var195",
        "Var196",
        "Var197",
        "Var198",
        "Var199",
        "Var200",
        "Var201",
        "Var202",
        "Var203",
        "Var204",
        "Var205",
        "Var206",
        "Var207",
        "Var208",
        "Var210",
        "Var212",
        "Var216",
        "Var217",
        "Var218",
        "Var219",
        "Var221",
        "Var223",
        "Var225",
        "Var226",
        "Var227",
        "Var228",
        "Var229",
    ]

    model = PerpetualBooster(
        categorical_features=categorical_features, iteration_limit=3, memory_limit=1
    )
    model.fit(X, y)


def test_tabarena_save_load():
    print("test_tabarena_save_load started.")
    X = _get_csv("tabarena_load_x.csv")
    y = _get_csv("tabarena_load_y.csv")

    categorical_features = [
        "position_-30",
        "position_-29",
        "position_-28",
        "position_-27",
        "position_-26",
        "position_-25",
        "position_-24",
        "position_-23",
        "position_-22",
        "position_-21",
        "position_-20",
        "position_-19",
        "position_-18",
        "position_-17",
        "position_-16",
        "position_-15",
        "position_-14",
        "position_-13",
        "position_-12",
        "position_-11",
        "position_-10",
        "position_-9",
        "position_-8",
        "position_-7",
        "position_-6",
        "position_-5",
        "position_-4",
        "position_-3",
        "position_-2",
        "position_-1",
        "position_1",
        "position_2",
        "position_3",
        "position_4",
        "position_5",
        "position_6",
        "position_7",
        "position_8",
        "position_9",
        "position_10",
        "position_11",
        "position_12",
        "position_13",
        "position_14",
        "position_15",
        "position_16",
        "position_17",
        "position_18",
        "position_19",
        "position_20",
        "position_21",
        "position_22",
        "position_23",
        "position_24",
        "position_25",
        "position_26",
        "position_27",
        "position_28",
        "position_29",
        "position_30",
    ]

    model = PerpetualBooster(
        categorical_features=categorical_features,
        memory_limit=3,
        num_threads=8,
        objective="LogLoss",
        iteration_limit=10,
        budget=2.0,
    )

    print(f"Starting fit... memory_limit={model.memory_limit}")
    model.fit(X, y)
    print("Fit completed.")
    print(f"Number of trees: {model.number_of_trees}")

    model_path = os.path.join(RESOURCES_DIR, "model.pkl")

    print(f"Saving model to {model_path}...")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print("Model saved.")

    print("Loading model...")
    with open(model_path, "rb") as f:
        loaded_model = pickle.load(f)
    print("Model loaded.")

    del loaded_model
    try:
        os.remove(model_path)
    except OSError:
        pass
