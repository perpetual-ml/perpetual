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
    budget = 1.0
    data_id = 46951  # 46951 -> Pumpkin Seeds, Else -> California Housing
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
seed: 0, mse: 0.19802932075312146, cpu time: 2.3125, wall time: 1.232259750366211
seed: 1, mse: 0.20229580459306082, cpu time: 3.1875, wall time: 1.3933138847351074
seed: 2, mse: 0.20418531802598813, cpu time: 3.671875, wall time: 1.700300931930542
seed: 3, mse: 0.19179947826902072, cpu time: 3.109375, wall time: 1.681952714920044
seed: 4, mse: 0.2065352390459141, cpu time: 2.671875, wall time: 1.1729986667633057
all mse: [0.19802932075312146, 0.20229580459306082, 0.20418531802598813, 0.19179947826902072, 0.2065352390459141]
avg mse: 0.20056903213742103
avg cpu time: 2.990625
avg wall time: 1.4145467281341553
cpu time / wall time: 2.1

budget=0.85, California Housing
seed: 0, mse: 0.19553129511198478, cpu time: 2.21875, wall time: 1.4096100330352783
seed: 1, mse: 0.19216813166982644, cpu time: 2.796875, wall time: 1.5446157455444336
seed: 2, mse: 0.20641167631066845, cpu time: 3.203125, wall time: 1.8815500736236572
seed: 3, mse: 0.18671747078429726, cpu time: 3.53125, wall time: 1.730372428894043
seed: 4, mse: 0.20073084238497707, cpu time: 2.40625, wall time: 1.4431695938110352
all mse: [0.19553129511198478, 0.19216813166982644, 0.20641167631066845, 0.18671747078429726, 0.20073084238497707]
avg mse: 0.1963118832523508
avg cpu time: 2.83125
avg wall time: 1.577873420715332
cpu time / wall time: 1.8

budget=1.15, California Housing
seed: 0, mse: 0.18733903559931364, cpu time: 4.671875, wall time: 2.790018320083618
seed: 1, mse: 0.18854020001840538, cpu time: 4.984375, wall time: 2.9729392528533936
seed: 2, mse: 0.19442923920930036, cpu time: 5.265625, wall time: 3.116692304611206
seed: 3, mse: 0.18098333241517914, cpu time: 5.578125, wall time: 3.285287380218506
seed: 4, mse: 0.19813965573232273, cpu time: 4.171875, wall time: 2.503753900527954
all mse: [0.18733903559931364, 0.18854020001840538, 0.19442923920930036, 0.18098333241517914, 0.19813965573232273]
avg mse: 0.18988629259490425
avg cpu time: 4.934375
avg wall time: 2.8872261524200438
cpu time / wall time: 1.7

budget=1.0, Pumpkin Seeds
seed: 0, roc_auc: 0.9466962305986696, cpu time: 1.5, wall time: 0.40040159225463867
seed: 1, roc_auc: 0.9443965790307255, cpu time: 2.125, wall time: 0.430269718170166
seed: 2, roc_auc: 0.9258720930232558, cpu time: 2.46875, wall time: 0.8848776817321777
seed: 3, roc_auc: 0.9560023310023311, cpu time: 1.796875, wall time: 0.43276190757751465
seed: 4, roc_auc: 0.9467079861816703, cpu time: 2.078125, wall time: 0.4988722801208496
all roc_auc: [0.9466962305986696, 0.9443965790307255, 0.9258720930232558, 0.9560023310023311, 0.9467079861816703]
avg roc_auc: 0.9439350439673305
avg cpu time: 1.99375
avg wall time: 0.5204818248748779
cpu time / wall time: 3.8
"""
