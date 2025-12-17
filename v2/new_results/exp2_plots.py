import os
import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR = "plots"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv("../capacity_results.csv")

num_cols = ["grid_size", "hot_fraction", "num_states", "num_actions",
            "success_rate", "train_time", "avg_eval_steps"]
for col in num_cols:
    df[col] = pd.to_numeric(df[col])

cache_tables = ["lru", "lfu"]
baseline_tables = ["array", "dict", "timestamp"]

grid_size = int(df["grid_size"].iloc[0])
env_type = df["env_type"].iloc[0]

baseline_df = df[df["table"].isin(baseline_tables)].set_index("table")

cache_colors = {
    "lru": "tab:blue",
    "lfu": "tab:orange",
}

baseline_colors = {
    "array": "tab:green",
    "dict": "tab:red",
    "timestamp": "tab:purple",
}

def plot_metric_vs_hot(metric: str,
                       ylabel: str,
                       title: str,
                       filename: str,
                       logy: bool = False):
    plt.figure(figsize=(8, 6))

    for table in cache_tables:
        sub = df[df["table"] == table].sort_values("hot_fraction")
        plt.plot(
            sub["hot_fraction"],
            sub[metric],
            marker="o",
            linestyle="-",
            color=cache_colors.get(table, None),
            label=table,
        )

    for base in baseline_tables:
        if base in baseline_df.index:
            y_val = baseline_df.loc[base, metric]
            plt.axhline(
                y=y_val,
                linestyle="--",
                linewidth=1.5,
                color=baseline_colors.get(base, None),
                alpha=0.9,
                label=f"{base} (baseline)",
            )

    plt.xlabel("Hot fraction (cache capacity / total |S×A|)")
    plt.ylabel(ylabel)
    plt.title(f"{title}\n(Grid {grid_size}×{grid_size}, env={env_type})")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(title="Q-table", loc="best")

    if logy:
        plt.yscale("log")

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, filename), dpi=200)
    plt.close()


plot_metric_vs_hot(
    metric="success_rate",
    ylabel="Success rate",
    title="Experiment 2: Success rate vs cache budget",
    filename="exp2_success_rate_vs_hot_fraction.png",
    logy=False,
)

plot_metric_vs_hot(
    metric="train_time",
    ylabel="Training time (seconds)",
    title="Experiment 2: Training time vs cache budget",
    filename="exp2_train_time_vs_hot_fraction.png",
    logy=True,
)

plot_metric_vs_hot(
    metric="avg_eval_steps",
    ylabel="Average evaluation steps",
    title="Experiment 2: Avg eval steps vs cache budget",
    filename="exp2_avg_steps_vs_hot_fraction.png",
    logy=False,
)

print("Saved plots in 'new_results/':")
print("  exp2_success_rate_vs_hot_fraction.png")
print("  exp2_train_time_vs_hot_fraction.png")
print("  exp2_avg_steps_vs_hot_fraction.png")
