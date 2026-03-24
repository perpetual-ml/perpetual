import argparse
import json
import re
import warnings
from functools import partial
from pathlib import Path
from time import process_time, time

import numpy as np
import optuna
import pandas as pd
import requests
from catboost import CatBoostClassifier, CatBoostRegressor
from perpetual import PerpetualBooster
from sklearn.datasets import fetch_openml
from sklearn.metrics import log_loss, mean_squared_error, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

# python package-python/examples/benchmark_comparison.py
# python package-python/examples/benchmark_comparison.py --no-optuna
# python package-python/examples/benchmark_comparison.py --skip-catboost
# python package-python/examples/benchmark_comparison.py --catboost-only
# python package-python/examples/benchmark_comparison.py --catboost-only --catboost-results-path tmp/catboost_results.csv

# https://www.openml.org/search?type=study&study_type=task&id=457&sort=tasks_included


"""
All PB (tuned) tabarena performance (v2.1.0):
                                   dataset problem_type   metric     method  metric_error  rank  total_methods dataset_id performance
                                       MIC   multiclass log_loss PB (tuned)      1.238202  18.0             18 46980
                             seismic-bumps       binary  roc_auc PB (tuned)      0.258542  18.0             18 46956      3-2
           taiwanese_bankruptcy_prediction       binary  roc_auc PB (tuned)      0.160871  18.0             18 46962      4-1
     students_dropout_and_academic_success   multiclass log_loss PB (tuned)      1.125762  18.0             18
                   Is-this-a-good-customer       binary  roc_auc PB (tuned)      0.279860  17.0             18
          blood-transfusion-service-center       binary  roc_auc PB (tuned)      0.277322  17.0             18 46913
                                     heloc       binary  roc_auc PB (tuned)      0.212887  17.0             18
                                    anneal   multiclass log_loss PB (tuned)      0.067830  17.0             18 46906
                                     churn       binary  roc_auc PB (tuned)      0.136474  17.0             18
                                  diabetes       binary  roc_auc PB (tuned)      0.175348  17.0             18 46921
                        kddcup09_appetency       binary  roc_auc PB (tuned)      0.369296  17.0             17
               coil2000_insurance_policies       binary  roc_auc PB (tuned)      0.267649  17.0             18
                                    splice   multiclass log_loss PB (tuned)      0.308573  17.0             18
                             Diabetes130US       binary  roc_auc PB (tuned)      0.411460  17.0             17
                    Amazon_employee_access       binary  roc_auc PB (tuned)      0.186921  17.0             17
                              Fitness_Club       binary  roc_auc PB (tuned)      0.194381  16.0             18
                                APSFailure       binary  roc_auc PB (tuned)      0.013972  16.0             17
HR_Analytics_Job_Change_of_Data_Scientists       binary  roc_auc PB (tuned)      0.210702  16.0             17
                     E-CommereShippingData       binary  roc_auc PB (tuned)      0.263426  16.0             18
                             hiva_agnostic   multiclass log_loss PB (tuned)      0.309314  16.0             17
                          website_phishing   multiclass log_loss PB (tuned)      0.294513  16.0             18
                        airfoil_self_noise   regression     rmse PB (tuned)      2.417587  16.0             18
                        QSAR_fish_toxicity   regression     rmse PB (tuned)      0.899660  15.0             18 46954
                 online_shoppers_intention       binary  roc_auc PB (tuned)      0.069808  15.0             18
                               qsar-biodeg       binary  roc_auc PB (tuned)      0.075866  15.0             18
               credit_card_clients_default       binary  roc_auc PB (tuned)      0.225365  15.0             17
                                    SDSS17   multiclass log_loss PB (tuned)      0.131018  15.0             17 46955
                               QSAR-TID-11   regression     rmse PB (tuned)      0.813204  15.0             17
                            bank-marketing       binary  roc_auc PB (tuned)      0.248461  15.0             17
                                  credit-g       binary  roc_auc PB (tuned)      0.218830  14.0             18 46918
                      maternal_health_risk   multiclass log_loss PB (tuned)      0.564076  14.0             18
                          GiveMeSomeCredit       binary  roc_auc PB (tuned)      0.142017  14.0             17 46929
                       Bank_Customer_Churn       binary  roc_auc PB (tuned)      0.139907  14.0             18
                        Marketing_Campaign       binary  roc_auc PB (tuned)      0.096731  14.0             18
                                       jm1       binary  roc_auc PB (tuned)      0.254839  13.0             18
                              wine_quality   regression     rmse PB (tuned)      0.629992  13.0             18
             concrete_compressive_strength   regression     rmse PB (tuned)      4.928130  13.0             18
                    physiochemical_protein   regression     rmse PB (tuned)      3.562011  13.0             17
               polish_companies_bankruptcy       binary  roc_auc PB (tuned)      0.063708  13.0             18
             healthcare_insurance_expenses   regression     rmse PB (tuned)   4625.943316  12.0             18
                             miami_housing   regression     rmse PB (tuned)  85758.029216  11.0             18
     hazelnut-spread-contaminant-detection       binary  roc_auc PB (tuned)      0.022465  11.0             18
          Another-Dataset-on-used-Fiat-500   regression     rmse PB (tuned)    746.753007  11.0             18
                         superconductivity   regression     rmse PB (tuned)      9.575404  11.0             17
          customer_satisfaction_in_airline       binary  roc_auc PB (tuned)      0.005844  11.0             17 46920
                        Food_Delivery_Time   regression     rmse PB (tuned)      7.488482  10.0             17
                                  diamonds   regression     rmse PB (tuned)    531.637650  10.0             17
                                    houses   regression     rmse PB (tuned)      0.217172  10.0             17
                              NATICUSdroid       binary  roc_auc PB (tuned)      0.014906   9.0             18
          in_vehicle_coupon_recommendation       binary  roc_auc PB (tuned)      0.166644   9.0             18
                               Bioresponse       binary  roc_auc PB (tuned)      0.123246   2.0             17
"""


# Dataset IDs with the most number of rows
# TASK_IDS = [46955, 46920, 46929]
# Dataset IDs with the least number of rows (max 1000 rows)
# TASK_IDS = [46913, 46921, 46906, 46954, 46918]
# All 11 datasets
TASK_IDS = [46913, 46921, 46906, 46954, 46918, 46955, 46920, 46929, 46980, 46956, 46962]

cb_iterations = 100
n_trials = 100
PERPETUAL_BUDGET = 2.0
use_optuna = True  # Set to False to use default CatBoost parameters (skip Optuna)
skip_catboost = False  # Set to True to skip CatBoost completely
n_splits = 3


def resolve_task(task_id):
    """
    Fetches the source dataset ID and target feature for a given OpenML task ID.
    """
    url = f"https://www.openml.org/api/v1/xml/task/{task_id}"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            text = response.text
            did_match = re.search(r"<oml:data_set_id>(\d+)</oml:data_set_id>", text)
            target_match = re.search(
                r"<oml:target_feature>(.*?)</oml:target_feature>", text
            )

            did = int(did_match.group(1)) if did_match else task_id
            target = target_match.group(1) if target_match else None
            return did, target
    except Exception as e:
        print(f"Warning: Could not resolve task {task_id}: {e}")
    return task_id, None


def prepare_data(task_id, seed):
    # Resolve Task ID to Dataset ID and target feature
    data_id, target = resolve_task(task_id)

    print(f"Loading Dataset ID {data_id} (from Task {task_id})...")
    try:
        # Fetch the dataset
        data = fetch_openml(
            data_id=data_id, return_X_y=True, as_frame=True, parser="auto"
        )
        X, y = data
        print(f"Data shape: {X.shape}")
    except Exception as e:
        print(f"Error fetching data for data_id {data_id}: {e}")
        raise

    # If y is a DataFrame (multi-target) or if we have a specific target name
    if isinstance(y, pd.DataFrame):
        if target and target in y.columns:
            y = y[target]
        else:
            y = y.iloc[:, 0]

    # Simple task type detection
    if (
        y.dtype == "object"
        or y.dtype.name == "category"
        or len(np.unique(y.dropna())) < 20
    ):
        n_classes = len(np.unique(y.dropna()))
        if n_classes == 2:
            task_type = "binary"
            scoring = "roc_auc"
            metric_function = roc_auc_score
        else:
            task_type = "multiclass"
            scoring = "neg_log_loss"
            metric_function = log_loss
    else:
        task_type = "regression"
        scoring = "neg_mean_squared_error"
        metric_function = mean_squared_error

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=seed,
        stratify=y if task_type in {"binary", "multiclass"} else None,
    )

    return (X_train, X_test, y_train, y_test, task_type, scoring, metric_function)


def build_cv(task_type, seed):
    if task_type in {"binary", "multiclass"}:
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return KFold(n_splits=n_splits, shuffle=True, random_state=seed)


def score_predictions(task_type, y_true, y_pred):
    if task_type == "binary":
        return roc_auc_score(y_true, y_pred)
    if task_type == "multiclass":
        return -log_loss(y_true, y_pred)
    return -mean_squared_error(y_true, y_pred)


def evaluate_predictions(task_type, y_true, y_pred):
    if task_type == "binary":
        return roc_auc_score(y_true, y_pred)
    if task_type == "multiclass":
        return log_loss(y_true, y_pred)
    return mean_squared_error(y_true, y_pred)


def predict_for_metric(model, X_data, task_type):
    if task_type == "binary":
        return model.predict_proba(X_data)[:, 1]
    if task_type == "multiclass":
        return model.predict_proba(X_data)
    return model.predict(X_data)


def objective_catboost(trial, X_train, y_train, task_type, seed):
    params = {
        "iterations": cb_iterations,
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.5, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-6, 10.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 1e-6, 10.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "od_type": "Iter",
        "od_wait": 20,
        "verbose": False,
        "random_seed": seed,
        "allow_writing_files": False,
    }

    # Identify cat features
    cat_features = X_train.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    cv = build_cv(task_type, seed)

    scores = []
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        if task_type in {"binary", "multiclass"}:
            model = CatBoostClassifier(**params)
        else:
            model = CatBoostRegressor(**params)

        model.fit(
            X_tr,
            y_tr,
            eval_set=(X_val, y_val),
            early_stopping_rounds=20,
            cat_features=cat_features,
        )

        preds = predict_for_metric(model, X_val, task_type)
        score = score_predictions(task_type, y_val, preds)
        scores.append(score)

    return np.mean(scores)


def run_catboost(X_train, y_train, X_test, y_test, task_type, seed):
    start_cpu = process_time()
    start_wall = time()

    if use_optuna:
        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        obj = partial(
            objective_catboost,
            X_train=X_train,
            y_train=y_train,
            task_type=task_type,
            seed=seed,
        )

        study.optimize(obj, n_trials=n_trials)
        best_params = study.best_params
        best_params["iterations"] = cb_iterations
    else:
        best_params = {}

    best_params["verbose"] = False
    best_params["random_seed"] = seed
    best_params["allow_writing_files"] = False

    cv = build_cv(task_type, seed)

    ensemble_preds = []
    cat_features = X_train.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    for train_idx, val_idx in cv.split(X_train, y_train):
        X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
        if task_type in {"binary", "multiclass"}:
            model = CatBoostClassifier(**best_params)
        else:
            model = CatBoostRegressor(**best_params)

        model.fit(X_tr, y_tr, cat_features=cat_features)

        ensemble_preds.append(predict_for_metric(model, X_test, task_type))

    y_pred = np.mean(ensemble_preds, axis=0)
    stop_cpu = process_time()
    stop_wall = time()

    metric = evaluate_predictions(task_type, y_test, y_pred)

    best_cv_score = study.best_value if use_optuna else None

    return {
        "metric": metric,
        "cpu_time": stop_cpu - start_cpu,
        "wall_time": stop_wall - start_wall,
        "best_cv_score": best_cv_score,
        "best_params": dict(best_params),
    }


def run_perpetual(X_train, y_train, X_test, y_test, task_type, seed):
    objective = "LogLoss" if task_type in {"binary", "multiclass"} else "SquaredLoss"
    model = PerpetualBooster(
        objective=objective,
        budget=PERPETUAL_BUDGET,
        max_cat=1000,
        seed=seed,
        iteration_limit=10000,
    )

    start_cpu = process_time()
    start_wall = time()
    model.fit(X_train, y_train)
    stop_cpu = process_time()
    stop_wall = time()

    y_pred = predict_for_metric(model, X_test, task_type)
    metric = evaluate_predictions(task_type, y_test, y_pred)

    return metric, stop_cpu - start_cpu, stop_wall - start_wall


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    parser = argparse.ArgumentParser(
        description=(
            "Benchmark PB vs CatBoost. Use --no-optuna to skip CatBoost tuning "
            "or --catboost-only to tune CatBoost and save its results without "
            "running PerpetualBooster."
        )
    )
    parser.add_argument(
        "--no-optuna",
        action="store_true",
        help="Skip CatBoost Optuna optimization and use default parameters",
    )
    parser.add_argument(
        "--skip-catboost",
        action="store_true",
        help="Skip CatBoost completely",
    )
    parser.add_argument(
        "--catboost-only",
        action="store_true",
        help=(
            "Run only CatBoost (with Optuna unless --no-optuna is set), skip "
            "PerpetualBooster, and save CatBoost results to CSV"
        ),
    )
    parser.add_argument(
        "--catboost-results-path",
        default=None,
        help=(
            "Optional CSV path for saved CatBoost results. Defaults to "
            "catboost_results.csv when --catboost-only is used. When used with --skip-catboost, reads results from this file to compare."
        ),
    )
    args = parser.parse_args()

    # Initialize execution flags with default values.
    use_optuna = True
    skip_catboost = False
    skip_perpetual = False

    if args.catboost_only and args.skip_catboost:
        parser.error("--catboost-only cannot be combined with --skip-catboost")

    if args.no_optuna:
        use_optuna = False
        print("Optuna optimization skipped. Using default CatBoost parameters.")

    if args.skip_catboost:
        skip_catboost = True
        print("CatBoost evaluation skipped completely.")

    catboost_results_path = None
    if args.catboost_only:
        skip_perpetual = True
        catboost_results_path = Path(
            args.catboost_results_path or "catboost_results.csv"
        )
        print(
            "Running CatBoost-only mode. PerpetualBooster evaluation skipped; "
            f"results will be saved to {catboost_results_path}."
        )
    elif args.catboost_results_path:
        catboost_results_path = Path(args.catboost_results_path)

    catboost_saved_rows = []

    loaded_catboost_results = {}
    if skip_catboost and catboost_results_path and catboost_results_path.exists():
        cb_df = pd.read_csv(catboost_results_path)
        for _, row in cb_df.iterrows():
            loaded_catboost_results[(int(row["task_id"]), int(row["seed"]))] = row[
                "metric"
            ]

    all_results = []

    for data_id in TASK_IDS:
        print(f"\nEvaluating Task ID: {data_id}")
        task_results = {"data_id": data_id}

        # Methodology: 5 seeds
        seeds = range(5)

        for seed in seeds:
            try:
                (
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    task_type,
                    scoring,
                    metric_function,
                ) = prepare_data(data_id, seed)

                # CatBoost
                if not skip_catboost:
                    cb_result = run_catboost(
                        X_train, y_train, X_test, y_test, task_type, seed
                    )
                    print(
                        "Seed "
                        f"{seed} - CatBoost: {cb_result['metric']:.4f}, CPU: "
                        f"{cb_result['cpu_time']:.2f}s, Wall: "
                        f"{cb_result['wall_time']:.2f}s"
                    )
                    if "CatBoost" not in task_results:
                        task_results["CatBoost"] = []
                    task_results["CatBoost"].append(cb_result["metric"])

                    if catboost_results_path is not None and not skip_catboost:
                        catboost_saved_rows.append(
                            {
                                "task_id": data_id,
                                "seed": seed,
                                "task_type": task_type,
                                "metric": cb_result["metric"],
                                "cpu_time": cb_result["cpu_time"],
                                "wall_time": cb_result["wall_time"],
                                "use_optuna": use_optuna,
                                "best_cv_score": cb_result["best_cv_score"],
                                "best_params": json.dumps(
                                    cb_result["best_params"], sort_keys=True
                                ),
                            }
                        )
                elif skip_catboost and (data_id, seed) in loaded_catboost_results:
                    cb_metric = loaded_catboost_results[(data_id, seed)]
                    if "CatBoost" not in task_results:
                        task_results["CatBoost"] = []
                    task_results["CatBoost"].append(cb_metric)

                # PerpetualBooster fixed-budget benchmark run.
                if not skip_perpetual:
                    pb_metric, pb_cpu, pb_wall = run_perpetual(
                        X_train, y_train, X_test, y_test, task_type, seed
                    )
                    print(
                        "Seed "
                        f"{seed} - Perpetual budget={PERPETUAL_BUDGET:.1f}: "
                        f"Test={pb_metric:.4f}, CPU: "
                        f"{pb_cpu:.2f}s, Wall: {pb_wall:.2f}s"
                    )

                    if "PB_budget" not in task_results:
                        task_results["PB_budget"] = []
                    task_results["PB_budget"].append(PERPETUAL_BUDGET)

                    if "PB_test" not in task_results:
                        task_results["PB_test"] = []
                    task_results["PB_test"].append(pb_metric)

            except Exception as e:
                print(f"Error evaluating Dataset ID {data_id} with seed {seed}: {e}")
                continue

        all_results.append(task_results)

    # Print final summary table
    print("\n" + "=" * 50)
    print("FINAL SUMMARY (Averaged over 5 seeds)")
    print("=" * 50)
    for res in all_results:
        summary = f"Dataset {res['data_id']}: "
        if "CatBoost" in res:
            summary += f"CatBoost={np.mean(res['CatBoost']):.4f}, "

        if "PB_test" in res:
            wins = None
            if "CatBoost" in res:
                wins = sum(pb > cb for pb, cb in zip(res["PB_test"], res["CatBoost"]))
            summary += f"PB(test)={np.mean(res['PB_test']):.4f}, PB(budget)=2.0"
            if wins is not None:
                summary += f", PB wins={wins}/5"

        print(summary)

    if catboost_results_path is not None and catboost_saved_rows:
        catboost_results_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(catboost_saved_rows).to_csv(catboost_results_path, index=False)
        print(f"Saved CatBoost results to {catboost_results_path}")

"""
==================================================
FINAL SUMMARY (Averaged over 5 seeds)
==================================================
Catboost with default parameters (no Optuna):
Dataset 46956: CatBoost=0.7690, PB_0.5=0.7269, PB_1.0=0.7531, PB_1.5=0.7525, PB_2.0=0.7530, PB_2.5=0.7563

((self.cfg.budget.max(1.0) * cols.max(1) as f32) * 10.0).round().clamp(256.0, ITER_LIMIT as f32) as usize
Dataset 46956:                  PB_0.5=0.7491, PB_1.0=0.7608, PB_1.5=0.7523, PB_2.0=0.7547, PB_2.5=0.7707

# DO NOT SET ITERATION_LIMIT
StructuralStopState - budget < 2.0:
Dataset 46956:                  PB_0.5=0.7491, PB_1.0=0.7608, PB_1.5=0.7523, PB_2.0=0.7624, PB_2.5=0.7713

# DO NOT SET ITERATION_LIMIT
StructuralStopState - budget < 0.5: 
Dataset 46956:                  PB_0.5=0.7521, PB_1.0=0.7593, PB_1.5=0.7561, PB_2.0=0.7624, PB_2.5=0.7713


Evaluating Task ID: 46956
Loading Dataset ID 46956 (from Task 46956)...
Data shape: (2584, 15)
Seed 0 - CatBoost: 0.7339, CPU: 5193.06s, Wall: 2657.54s
Seed 0 - Perpetual CV (budget=0.5): 0.7519
Seed 0 - Perpetual CV (budget=1.0): 0.7721
Seed 0 - Perpetual CV (budget=1.5): 0.7736
Seed 0 - Perpetual CV (budget=2.0): 0.7720
Seed 0 - Perpetual CV (budget=2.5): 0.7708
Seed 0 - Perpetual best budget=1.5: CV=0.7736, Test=0.6891, CPU: 17.52s, Wall: 6.78s
Loading Dataset ID 46956 (from Task 46956)...
Data shape: (2584, 15)
Seed 1 - CatBoost: 0.7813, CPU: 3778.50s, Wall: 2313.68s
Seed 1 - Perpetual CV (budget=0.5): 0.7270
Seed 1 - Perpetual CV (budget=1.0): 0.7411
Seed 1 - Perpetual CV (budget=1.5): 0.7394
Seed 1 - Perpetual CV (budget=2.0): 0.7500
Seed 1 - Perpetual CV (budget=2.5): 0.7457
Seed 1 - Perpetual best budget=2.0: CV=0.7500, Test=0.7895, CPU: 18.23s, Wall: 6.19s
Loading Dataset ID 46956 (from Task 46956)...
Data shape: (2584, 15)
Seed 2 - CatBoost: 0.7926, CPU: 3518.80s, Wall: 2120.20s
Seed 2 - Perpetual CV (budget=0.5): 0.7430
Seed 2 - Perpetual CV (budget=1.0): 0.7671
Seed 2 - Perpetual CV (budget=1.5): 0.7611
Seed 2 - Perpetual CV (budget=2.0): 0.7641
Seed 2 - Perpetual CV (budget=2.5): 0.7719
Seed 2 - Perpetual best budget=2.5: CV=0.7719, Test=0.8183, CPU: 19.28s, Wall: 6.75s
Loading Dataset ID 46956 (from Task 46956)...
Data shape: (2584, 15)
Seed 3 - CatBoost: 0.7282, CPU: 3405.52s, Wall: 1843.29s
Seed 3 - Perpetual CV (budget=0.5): 0.7425
Seed 3 - Perpetual CV (budget=1.0): 0.7713
Seed 3 - Perpetual CV (budget=1.5): 0.7685
Seed 3 - Perpetual CV (budget=2.0): 0.7711
Seed 3 - Perpetual CV (budget=2.5): 0.7689
Seed 3 - Perpetual best budget=1.0: CV=0.7713, Test=0.7717, CPU: 19.77s, Wall: 6.93s
Loading Dataset ID 46956 (from Task 46956)...
Data shape: (2584, 15)
Seed 4 - CatBoost: 0.7539, CPU: 2376.50s, Wall: 1441.93s
Seed 4 - Perpetual CV (budget=0.5): 0.7460
Seed 4 - Perpetual CV (budget=1.0): 0.7610
Seed 4 - Perpetual CV (budget=1.5): 0.7658
Seed 4 - Perpetual CV (budget=2.0): 0.7744
Seed 4 - Perpetual CV (budget=2.5): 0.7731
Seed 4 - Perpetual best budget=2.0: CV=0.7744, Test=0.7538, CPU: 19.03s, Wall: 5.77s

==================================================
FINAL SUMMARY (Averaged over 5 seeds)
==================================================
Dataset 46956: CatBoost=0.7580, PB(test)=0.7645, PB(CV score)=0.7683, PB(best budget avg)=1.80


Evaluating Task ID: 46962
Loading Dataset ID 46962 (from Task 46962)...
Data shape: (6819, 94)
Seed 0 - CatBoost: 0.9294, CPU: 26482.53s, Wall: 4417.47s
Seed 0 - Perpetual CV (budget=0.5): 0.9315
Seed 0 - Perpetual CV (budget=1.0): 0.9507
Seed 0 - Perpetual CV (budget=1.5): 0.9503
Seed 0 - Perpetual CV (budget=2.0): 0.9501
Seed 0 - Perpetual CV (budget=2.5): 0.9484
Seed 0 - Perpetual best budget=1.0: CV=0.9507, Test=0.9338, CPU: 13021.89s, Wall: 2424.28s
Loading Dataset ID 46962 (from Task 46962)...
Data shape: (6819, 94)
Seed 1 - CatBoost: 0.9528, CPU: 8262.08s, Wall: 1073.73s
Seed 1 - Perpetual CV (budget=0.5): 0.9317
Seed 1 - Perpetual CV (budget=1.0): 0.9382
Seed 1 - Perpetual CV (budget=1.5): 0.9379
Seed 1 - Perpetual CV (budget=2.0): 0.9425
Seed 1 - Perpetual CV (budget=2.5): 0.9391
Seed 1 - Perpetual best budget=2.0: CV=0.9425, Test=0.9535, CPU: 7514.67s, Wall: 1250.49s
Loading Dataset ID 46962 (from Task 46962)...
Data shape: (6819, 94)
Seed 2 - CatBoost: 0.9546, CPU: 8852.00s, Wall: 1229.24s
Seed 2 - Perpetual CV (budget=0.5): 0.9225
Seed 2 - Perpetual CV (budget=1.0): 0.9381
Seed 2 - Perpetual CV (budget=1.5): 0.9387
Seed 2 - Perpetual CV (budget=2.0): 0.9389
Seed 2 - Perpetual CV (budget=2.5): 0.9378
Seed 2 - Perpetual CV (budget=2.5): 0.9378
Seed 2 - Perpetual best budget=2.0: CV=0.9389, Test=0.9590, CPU: 9437.67s, Wall: 1530.59s
Loading Dataset ID 46962 (from Task 46962)...
Data shape: (6819, 94)
Seed 3 - CatBoost: 0.9359, CPU: 13050.34s, Wall: 1737.22s
Seed 3 - Perpetual CV (budget=0.5): 0.9196
Seed 3 - Perpetual CV (budget=1.0): 0.9444
Seed 3 - Perpetual CV (budget=1.5): 0.9426
Seed 3 - Perpetual CV (budget=2.0): 0.9445
Seed 3 - Perpetual CV (budget=2.5): 0.9436
Seed 3 - Perpetual best budget=2.0: CV=0.9445, Test=0.9298, CPU: 10488.19s, Wall: 1712.49s
Loading Dataset ID 46962 (from Task 46962)...
Data shape: (6819, 94)
Seed 4 - CatBoost: 0.9444, CPU: 67732.55s, Wall: 10211.18s
Seed 4 - Perpetual CV (budget=0.5): 0.9271
Seed 4 - Perpetual CV (budget=1.0): 0.9441
Seed 4 - Perpetual CV (budget=1.5): 0.9381
Seed 4 - Perpetual CV (budget=2.0): 0.9379
Seed 4 - Perpetual CV (budget=2.5): 0.9368
Seed 4 - Perpetual best budget=1.0: CV=0.9441, Test=0.9458, CPU: 9515.03s, Wall: 2417.26s

==================================================
FINAL SUMMARY (Averaged over 5 seeds)
==================================================
Dataset 46962: CatBoost=0.9434, PB(test)=0.9444, PB(CV score)=0.9441, PB(best budget avg)=1.60



row subsampling and max_cat=1000:

Evaluating Task ID: 46955
Loading Dataset ID 46955 (from Task 46955)...
Data shape: (78053, 11)
Seed 0 - CatBoost: 0.0769, CPU: 16382.72s, Wall: 3170.04s
Seed 0 - Perpetual CV (budget=0.5): -0.1410
Seed 0 - Perpetual CV (budget=1.0): -0.1125
Seed 0 - Perpetual CV (budget=1.5): -0.1413
Seed 0 - Perpetual CV (budget=2.0): -0.3378
Seed 0 - Perpetual CV (budget=2.5): -0.8828
Seed 0 - Perpetual best budget=1.0: CV=-0.1125, Test=0.1134, CPU: 152.23s, Wall: 50.51s


no row subsampling and max_cat=9000:

Evaluating Task ID: 46955
Loading Dataset ID 46955 (from Task 46955)...
Data shape: (78053, 11)
Seed 0 - CatBoost: 0.0769, CPU: 23710.47s, Wall: 4679.17s
Seed 0 - Perpetual CV (budget=0.5): -0.3614
Seed 0 - Perpetual CV (budget=1.0): -0.2878
Seed 0 - Perpetual CV (budget=1.5): -0.2289
Seed 0 - Perpetual CV (budget=2.0): -0.1945
Seed 0 - Perpetual CV (budget=2.5): -0.1684
Seed 0 - Perpetual best budget=2.5: CV=-0.1684, Test=0.1782, CPU: 1871.59s, Wall: 1351.24s
Loading Dataset ID 46955 (from Task 46955)...
Data shape: (78053, 11)


no row subsampling and max_cat=1000:

Evaluating Task ID: 46955
Loading Dataset ID 46955 (from Task 46955)...
Data shape: (78053, 11)
Seed 0 - CatBoost: 0.0769, CPU: 20905.67s, Wall: 3693.78s
Seed 0 - Perpetual CV (budget=0.5): -0.1410
Seed 0 - Perpetual CV (budget=1.0): -0.1190
Seed 0 - Perpetual CV (budget=1.5): -0.1037
Seed 0 - Perpetual CV (budget=2.0): -0.1687
Seed 0 - Perpetual CV (budget=2.5): -0.1788
Seed 0 - Perpetual best budget=1.5: CV=-0.1037, Test=0.1558, CPU: 325.73s, Wall: 167.95s
"""
