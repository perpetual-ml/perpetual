import optuna
import numpy as np
from time import process_time, time
from functools import partial
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.metrics import mean_squared_error, log_loss
from sklearn.datasets import fetch_covtype, fetch_california_housing
from sklearn.model_selection import train_test_split, cross_validate


def prepare_data(cal_housing, seed):
    if cal_housing:
        data, target = fetch_california_housing(return_X_y=True, as_frame=True)
        scoring = "neg_mean_squared_error"
        metric_function = mean_squared_error
        metric_name = "mse"
        LGBMBooster = LGBMRegressor
    else:
        data, target = fetch_covtype(return_X_y=True, as_frame=True)
        scoring = "neg_log_loss"
        metric_function = log_loss
        metric_name = "log_loss"
        LGBMBooster = LGBMClassifier
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
    cal_housing = True  # True -> California Housing, False -> Cover Types
    n_estimators = 100
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
        ) = prepare_data(cal_housing, seed)

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
        else:
            y_pred = np.mean([model.predict(X_test) for model in models], axis=0)
        metric = metric_function(y_test, y_pred)
        metrics.append(metric)

        print(f"seed: {seed}, cpu time: {stop - start}, {metric_name}: {metric}")

    print(f"avg cpu time: {np.mean(cpu_times)}, avg {metric_name}: {np.mean(metrics)}")
    print(f"avg wall time: {np.mean(wall_times)}")
    print(f"cpu time / wall time: {(np.mean(cpu_times)/np.mean(wall_times)):.1f}")
