import numpy as np
from time import process_time
from perpetual import PerpetualBooster
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, log_loss
from sklearn.datasets import fetch_covtype, fetch_california_housing


def prepare_data(cal_housing, seed):
    if cal_housing:
        data, target = fetch_california_housing(return_X_y=True, as_frame=True)
        metric_function = mean_squared_error
        metric_name = "mse"
        objective = "SquaredLoss"
    else:
        data, target = fetch_covtype(return_X_y=True, as_frame=True)
        metric_function = log_loss
        metric_name = "log_loss"
        objective = "LogLoss"
    X_train, X_test, y_train, y_test = train_test_split(
        data, target, test_size=0.2248, random_state=seed
    )
    return X_train, X_test, y_train, y_test, metric_function, metric_name, objective


if __name__ == "__main__":
    budget = 1.5
    cal_housing = True
    cpu_times = []
    metrics = []
    for seed in [0, 1, 2, 3, 4]:
        X_train, X_test, y_train, y_test, metric_function, metric_name, objective = (
            prepare_data(cal_housing, seed)
        )
        model = PerpetualBooster(objective=objective)
        start = process_time()
        model.fit(X_train, y_train, budget=budget)
        stop = process_time()
        cpu_times.append(stop - start)

        if metric_name == "log_loss":
            y_pred = model.predict_proba(X_test)
        else:
            y_pred = model.predict(X_test)
        metric = metric_function(y_test, y_pred)
        metrics.append(metric)

        print(
            f"seed: {seed}, cpu time: {stop - start}, {metric_name}: {metric}, n_trees: {model.number_of_trees}"
        )

    print(
        f"average cpu time: {np.mean(cpu_times)}, average {metric_name}: {np.mean(metrics)}"
    )
