from importlib.metadata import version
from time import process_time, time

import numpy as np
from perpetual import PerpetualBooster
from sklearn.datasets import fetch_california_housing, fetch_openml
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split

# uv run python package-python/examples/benchmark_perpetual.py


def prepare_data(data_id, seed):
    if data_id == 46951:
        data, target = fetch_openml(data_id=data_id, return_X_y=True, as_frame=True)
        metric_function = roc_auc_score
        metric_name = "roc_auc"
        objective = "LogLoss"
    else:
        data, target = fetch_california_housing(return_X_y=True, as_frame=True)
        metric_function = mean_squared_error
        metric_name = "mse"
        objective = "SquaredLoss"

    X_train, X_test, y_train, y_test = train_test_split(
        data, target, test_size=0.2248, random_state=seed
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        metric_function,
        metric_name,
        objective,
    )


if __name__ == "__main__":
    print(f"perpetual: {version('perpetual')}")
    budget = 1.15
    data_id = ""  # 46951 -> Pumpkin Seeds, Else -> California Housing
    cpu_times = []
    wall_times = []
    metrics = []

    for seed in range(5):
        (
            X_train,
            X_test,
            y_train,
            y_test,
            metric_function,
            metric_name,
            objective,
        ) = prepare_data(data_id, seed)

        model = PerpetualBooster(
            objective=objective, budget=budget, iteration_limit=10000
        )

        start = process_time()
        tick = time()
        model.fit(X_train, y_train)
        stop = process_time()
        cpu_times.append(stop - start)
        wall_times.append(time() - tick)

        if metric_name == "log_loss":
            y_pred = model.predict_proba(X_test)
        elif metric_name == "roc_auc":
            y_pred = model.predict_proba(X_test)[:, 1]
        else:
            y_pred = model.predict(X_test)
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
budget=0.76, California Housing
seed: 0, mse: 0.19802932075312182, cpu time: 15.546875, wall time: 2.581190824508667
seed: 1, mse: 0.20228360030470313, cpu time: 16.96875, wall time: 2.6180028915405273
seed: 2, mse: 0.2076308801655241, cpu time: 17.015625, wall time: 2.604408025741577
seed: 3, mse: 0.19179947832029257, cpu time: 21.390625, wall time: 3.043393135070801
seed: 4, mse: 0.20654010702343878, cpu time: 14.1875, wall time: 2.307356595993042
all mse: [0.19802932075312182, 0.20228360030470313, 0.2076308801655241, 0.19179947832029257, 0.20654010702343878]
avg mse: 0.20125667731341607
avg cpu time: 17.021875
avg wall time: 2.6139562129974365
cpu time / wall time: 6.5

budget=0.85, California Housing
seed: 0, mse: 0.19554654672820126, cpu time: 17.265625, wall time: 2.841470956802368
seed: 1, mse: 0.1921681316698263, cpu time: 18.875, wall time: 2.753267526626587
seed: 2, mse: 0.20640364072185663, cpu time: 23.28125, wall time: 3.397728443145752
seed: 3, mse: 0.18671747074888528, cpu time: 21.921875, wall time: 3.1276187896728516
seed: 4, mse: 0.20073759266237517, cpu time: 18.15625, wall time: 2.80734920501709
all mse: [0.19554654672820126, 0.1921681316698263, 0.20640364072185663, 0.18671747074888528, 0.20073759266237517]
avg mse: 0.1963146765062289
avg cpu time: 19.9
avg wall time: 2.9649657249450683
cpu time / wall time: 6.7

budget=1.15, California Housing
seed: 0, mse: 0.1873390355985955, cpu time: 39.078125, wall time: 5.320531845092773
seed: 1, mse: 0.18854382758341648, cpu time: 39.9375, wall time: 5.306451320648193
seed: 2, mse: 0.19442165522078822, cpu time: 39.546875, wall time: 5.336567640304565
seed: 3, mse: 0.18098408278530434, cpu time: 39.765625, wall time: 5.3054115772247314
seed: 4, mse: 0.19816332648421614, cpu time: 30.703125, wall time: 4.285067558288574
all mse: [0.1873390355985955, 0.18854382758341648, 0.19442165522078822, 0.18098408278530434, 0.19816332648421614]
avg mse: 0.1898903855344641
avg cpu time: 37.80625
avg wall time: 5.077028608322143
cpu time / wall time: 7.4

budget=1.0, Pumpkin Seeds
seed: 0, roc_auc: 0.946683560342097, cpu time: 1.890625, wall time: 1.281526803970337
seed: 1, roc_auc: 0.9443965790307254, cpu time: 2.046875, wall time: 1.1355137825012207
seed: 2, roc_auc: 0.9258593431252551, cpu time: 6.203125, wall time: 1.560915470123291
seed: 3, roc_auc: 0.956002331002331, cpu time: 1.90625, wall time: 1.0808000564575195
seed: 4, roc_auc: 0.9465936801463116, cpu time: 2.015625, wall time: 1.1039233207702637
all roc_auc: [0.946683560342097, 0.9443965790307254, 0.9258593431252551, 0.956002331002331, 0.9465936801463116]
avg roc_auc: 0.9439070987293441
avg cpu time: 2.8125
avg wall time: 1.2262436866760253
cpu time / wall time: 2.3
"""
