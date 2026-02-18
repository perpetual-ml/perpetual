import os

import matplotlib.pyplot as plt
import pandas as pd


def plot_benchmark():
    if not os.path.exists("results.csv"):
        print("results.csv not found.")
        return

    df = pd.read_csv("results.csv")

    # Calculate Total Step Time
    df["StepTimeMs"] = df["Time_Fit_ms"] + df["Time_Prune_ms"]

    # Calculate Cumulative Time per strategy
    df["CumulativeTimeMs"] = df.groupby("Strategy")["StepTimeMs"].cumsum()

    # Setup styles
    plt.style.use("ggplot")
    # 2x2 Grid
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    strategies = df["Strategy"].unique()

    # 1. Cumulative Time
    ax = axes[0]
    for strategy in strategies:
        subset = df[df["Strategy"] == strategy]
        ax.plot(
            subset["Batch"],
            subset["CumulativeTimeMs"] / 1000.0,
            marker="o",
            label=strategy,
        )
    ax.set_title("Cumulative Training + Pruning Time")
    ax.set_xlabel("Batch Index")
    ax.set_ylabel("Time (seconds)")
    ax.legend()

    # 2. Step Time
    ax = axes[1]
    for strategy in strategies:
        subset = df[df["Strategy"] == strategy]
        ax.plot(subset["Batch"], subset["StepTimeMs"], marker="o", label=strategy)
    ax.set_title("Step Time (Fit + Prune)")
    ax.set_xlabel("Batch Index")
    ax.set_ylabel("Time (ms)")
    ax.legend()

    # 3. MSE
    ax = axes[2]
    # Highlight No Pruning and Statistical as they were interesting
    for strategy in strategies:
        subset = df[df["Strategy"] == strategy]
        ax.plot(subset["Batch"], subset["MSE"], marker="o", label=strategy)
    ax.set_title("Test MSE (Cumulative Data)")
    ax.set_xlabel("Batch Index")
    ax.set_ylabel("Mean Squared Error")
    ax.set_yscale("log")  # Log scale because No Pruning MSE grew very large
    ax.legend()

    # 4. Nodes
    ax = axes[3]
    for strategy in strategies:
        subset = df[df["Strategy"] == strategy]
        ax.plot(subset["Batch"], subset["Nodes"], marker="o", label=strategy)
    ax.set_title("Total Model Nodes")
    ax.set_xlabel("Batch Index")
    ax.set_ylabel("Count")
    ax.legend()

    plt.tight_layout()
    plt.savefig("benchmark_results.png")
    print("Graph saved to benchmark_results.png")


if __name__ == "__main__":
    plot_benchmark()
