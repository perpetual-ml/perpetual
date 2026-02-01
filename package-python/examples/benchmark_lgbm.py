from functools import partial
from time import process_time, time

import numpy as np
import optuna
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.datasets import fetch_california_housing, fetch_openml
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import cross_validate, train_test_split

# uv run python package-python/examples/benchmark_lgbm.py


def prepare_data(data_id, seed):
    if data_id == 46951:
        n_estimators = 100
        data, target = fetch_openml(data_id=data_id, return_X_y=True, as_frame=True)
        scoring = "neg_log_loss"
        metric_function = roc_auc_score
        metric_name = "roc_auc"
        LGBMBooster = LGBMClassifier
    else:
        n_estimators = 200
        data, target = fetch_california_housing(return_X_y=True, as_frame=True)
        scoring = "neg_mean_squared_error"
        metric_function = mean_squared_error
        metric_name = "mse"
        LGBMBooster = LGBMRegressor

    X_train, X_test, y_train, y_test = train_test_split(
        data, target, test_size=0.2248, random_state=seed
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        scoring,
        metric_function,
        metric_name,
        LGBMBooster,
        n_estimators,
    )


best_cv_results = None
cv_results = None


def save_best_cv_results(study, trial):
    global best_cv_results
    if study.best_trial.number == trial.number:
        best_cv_results = cv_results


def objective_function(
    trial, seed, n_estimators, LGBMBooster, X_train, y_train, scoring
):
    global cv_results
    params = {
        "seed": seed,
        "verbosity": -1,
        "n_estimators": n_estimators,
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.5, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 1e-6, 1.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-6, 1.0, log=True),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
        "subsample": trial.suggest_float("subsample", 0.2, 1.0),
        "subsample_freq": trial.suggest_int("subsample_freq", 1, 10),
        "max_depth": trial.suggest_int("max_depth", 3, 33),
        "num_leaves": trial.suggest_int("num_leaves", 2, 1024),
        "min_child_samples": trial.suggest_int("min_child_samples", 1, 100),
    }
    model = LGBMBooster(**params)
    cv_results = cross_validate(
        model,
        X_train,
        y_train,
        cv=5,
        scoring=scoring,
        return_train_score=True,
        return_estimator=True,
    )
    return -1 * np.mean(cv_results["test_score"])


if __name__ == "__main__":
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    data_id = ""  # 46951 -> Pumpkin Seeds, Else -> California Housing
    n_trials = 100
    cpu_times = []
    wall_times = []
    metrics = []

    for seed in range(5):
        (
            X_train,
            X_test,
            y_train,
            y_test,
            scoring,
            metric_function,
            metric_name,
            LGBMBooster,
            n_estimators,
        ) = prepare_data(data_id, seed)

        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction="minimize", sampler=sampler)

        obj = partial(
            objective_function,
            seed=seed,
            n_estimators=n_estimators,
            LGBMBooster=LGBMBooster,
            X_train=X_train,
            y_train=y_train,
            scoring=scoring,
        )

        start = process_time()
        tick = time()
        study.optimize(obj, n_trials=n_trials, callbacks=[save_best_cv_results])
        stop = process_time()
        cpu_times.append(stop - start)
        wall_times.append(time() - tick)

        models = best_cv_results["estimator"]
        if metric_name == "log_loss":
            y_pred = np.mean([model.predict_proba(X_test) for model in models], axis=0)
        elif metric_name == "roc_auc":
            y_pred = np.mean(
                [model.predict_proba(X_test)[:, 1] for model in models], axis=0
            )
        else:
            y_pred = np.mean([model.predict(X_test) for model in models], axis=0)
        metric = metric_function(y_test, y_pred)
        metrics.append(metric)

        print(
            f"seed: {seed}, {metric_name}: {metric}, cpu time: {stop - start}, wall time: {time() - tick}"
        )

    print(f"all {metric_name}: {metrics}")
    print(f"avg {metric_name}: {np.mean(metrics)}")
    print(f"avg cpu time: {np.mean(cpu_times)}")
    print(f"avg wall time: {np.mean(wall_times)}")
    print(f"cpu time / wall time: {(np.mean(cpu_times) / np.mean(wall_times)):.1f}")

"""
n_estimators = 50, California Housing
seed: 0, mse: 0.19937027467135718, cpu time: 755.828125, wall time: 82.59403848648071
seed: 1, mse: 0.20512843278161733, cpu time: 836.890625, wall time: 86.49860548973083
seed: 2, mse: 0.2111257435319618, cpu time: 1021.953125, wall time: 107.5655300617218
seed: 3, mse: 0.18901271302872738, cpu time: 1036.40625, wall time: 105.57267260551453
seed: 4, mse: 0.2027074725309233, cpu time: 1232.15625, wall time: 125.44857382774353
all mse: [0.19937027467135718, 0.20512843278161733, 0.2111257435319618, 0.18901271302872738, 0.2027074725309233]
avg mse: 0.2014689273089174
avg cpu time: 976.646875
avg wall time: 101.49670634269714
cpu time / wall time: 9.6

n_estimators = 100, California Housing
seed: 0, mse: 0.19177274620659984, cpu time: 1374.625, wall time: 145.1648759841919
seed: 1, mse: 0.19798397157401768, cpu time: 1965.40625, wall time: 199.64547324180603
seed: 2, mse: 0.2029016710461026, cpu time: 2001.125, wall time: 203.88489294052124
seed: 3, mse: 0.18957536463140065, cpu time: 1963.046875, wall time: 200.34187197685242
seed: 4, mse: 0.19931639682607546, cpu time: 1370.125, wall time: 139.70405340194702
all mse: [0.19177274620659984, 0.19798397157401768, 0.2029016710461026, 0.18957536463140065, 0.19931639682607546]
avg mse: 0.19631003005683925
avg cpu time: 1734.865625
avg wall time: 177.68452434539796
cpu time / wall time: 9.8

n_estimators = 200, California Housing
seed: 0, mse: 0.19010826010166118, cpu time: 8956.765625, wall time: 1013.7352392673492
seed: 1, mse: 0.1902000284676032, cpu time: 11342.203125, wall time: 1325.3336989879608
seed: 2, mse: 0.1917257514154116, cpu time: 6537.421875, wall time: 788.6310698986053
seed: 3, mse: 0.18256633488990814, cpu time: 12188.453125, wall time: 1628.9395711421967
seed: 4, mse: 0.19588043620958964, cpu time: 9952.375, wall time: 1096.520533323288
all mse: [0.19010826010166118, 0.1902000284676032, 0.1917257514154116, 0.18256633488990814, 0.19588043620958964]
avg mse: 0.19009616221683473
avg cpu time: 9795.44375
avg wall time: 1170.2897990703582
cpu time / wall time: 8.4

n_estimators = 100, Pumpkin_Seeds
seed: 0, roc_auc: 0.9504212860310421, cpu time: 357.734375, wall time: 42.51193451881409
seed: 1, roc_auc: 0.948596769084574, cpu time: 351.046875, wall time: 54.16120266914368
seed: 2, roc_auc: 0.9275040799673603, cpu time: 445.390625, wall time: 61.674174070358276
seed: 3, roc_auc: 0.9543680956724435, cpu time: 331.125, wall time: 37.69600176811218
seed: 4, roc_auc: 0.9433550091444829, cpu time: 350.65625, wall time: 41.92474722862244
all roc_auc: [0.9504212860310421, 0.948596769084574, 0.9275040799673603, 0.9543680956724435, 0.9433550091444829]
avg roc_auc: 0.9448490479799805
avg cpu time: 367.190625
avg wall time: 47.57699012756348
cpu time / wall time: 7.7
"""
