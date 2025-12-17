import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR = "plots"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv("../scaling_results.csv")

for col in ["grid_size", "num_states", "num_actions",
            "success_rate", "train_time", "avg_eval_steps"]:
    df[col] = pd.to_numeric(df[col])

tables = sorted(df["table"].unique())
grid_sizes = sorted(df["grid_size"].unique())


def plot_metric(metric: str,
                ylabel: str,
                title: str,
                filename: str,
                logy: bool = False):
    plt.figure(figsize=(8, 6))

    for table in tables:
        sub = df[df["table"] == table].sort_values("grid_size")
        plt.plot(
            sub["grid_size"],
            sub[metric],
            marker="o",
            linestyle="-",
            label=table,
        )

    plt.xlabel("Grid size (N × N)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(title="Q-table", loc="best")

    ax = plt.gca()

    if logy:
        y_vals = df[metric].to_numpy()
        y_min = float(np.min(y_vals))
        y_max = float(np.max(y_vals))

        y_min *= 0.9
        y_max *= 1.1

        log_min = np.log10(y_min)
        log_max = np.log10(y_max)

        num_intervals = 5
        num_ticks = num_intervals + 1

        ticks = np.logspace(log_min, log_max, num=num_ticks)
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{t:.1f}" for t in ticks])

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, filename), dpi=200)
    plt.close()


plot_metric(
    metric="success_rate",
    ylabel="Success rate",
    title="Experiment 1: Success rate vs Grid size",
    filename="exp1_success_rate_vs_grid_size.png",
    logy=False,
)

plot_metric(
    metric="train_time",
    ylabel="Training time (seconds)",
    title="Experiment 1: Training time vs Grid size",
    filename="exp1_train_time_vs_grid_size.png",
    logy=True,
)

plot_metric(
    metric="avg_eval_steps",
    ylabel="Average evaluation steps",
    title="Experiment 1: Avg eval steps vs Grid size",
    filename="exp1_avg_steps_vs_grid_size.png",
    logy=False,
)

print("Saved plots in 'new_results/':")
print("  exp1_success_rate_vs_grid_size.png")
print("  exp1_train_time_vs_grid_size.png")
print("  exp1_avg_steps_vs_grid_size.png")
