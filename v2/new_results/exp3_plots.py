import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def rolling_mean(series, window):
    if window <= 1:
        return series
    return series.rolling(window=window, min_periods=1).mean()

def main(csv_path="../cache_stats_exp3.csv", smooth_window=50):
    df = pd.read_csv(csv_path)

    for col in ["episode", "episode_length",
                "promotions_to_hot", "demotions_to_cold"]:
        df[col] = pd.to_numeric(df[col])

    df["promo_per_step"] = df["promotions_to_hot"] / df["episode_length"].replace(0, np.nan)
    df["demo_per_step"] = df["demotions_to_cold"] / df["episode_length"].replace(0, np.nan)

    tables = sorted(df["table"].unique())

    plt.figure(figsize=(10, 6))
    for table in tables:
        sub = df[df["table"] == table].sort_values("episode")
        y = rolling_mean(sub["promo_per_step"], smooth_window)
        plt.plot(sub["episode"], y, label=table)
    plt.xlabel("Episode")
    plt.ylabel(f"Promotions per step (rolling mean, window={smooth_window})")
    plt.title("Experiment 3: Promotions to hot cache per step")
    plt.yscale("log")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(title="Q-table", loc="best")
    plt.tight_layout()
    plt.savefig("exp3_promos_per_step_log.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6))
    for table in tables:
        sub = df[df["table"] == table].sort_values("episode")
        y = rolling_mean(sub["demo_per_step"], smooth_window)
        plt.plot(sub["episode"], y, label=table)
    plt.xlabel("Episode")
    plt.ylabel(f"Demotions per step (rolling mean, window={smooth_window})")
    plt.title("Experiment 3: Demotions from hot cache per step")
    plt.yscale("log")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(title="Q-table", loc="best")
    plt.tight_layout()
    plt.savefig("exp3_demos_per_step_log.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6))
    for table in tables:
        sub = df[df["table"] == table].sort_values("episode")
        y = rolling_mean(sub["promotions_to_hot"], smooth_window)
        plt.plot(sub["episode"], y, label=table)
    plt.xlabel("Episode")
    plt.ylabel(f"Promotions to hot (rolling mean, window={smooth_window})")
    plt.title("Experiment 3: Promotions to hot cache per episode")
    plt.yscale("log")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(title="Q-table", loc="best")
    plt.tight_layout()
    plt.savefig("exp3_promotions_per_episode.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6))
    for table in tables:
        sub = df[df["table"] == table].sort_values("episode")
        y = rolling_mean(sub["demotions_to_cold"], smooth_window)
        plt.plot(sub["episode"], y, label=table)
    plt.xlabel("Episode")
    plt.ylabel(f"Demotions to cold (rolling mean, window={smooth_window})")
    plt.title("Experiment 3: Demotions from hot cache per episode")
    plt.yscale("log")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(title="Q-table", loc="best")
    plt.tight_layout()
    plt.savefig("exp3_demotions_per_episode.png", dpi=200)
    plt.close()

    print("Saved plots:")
    print("  exp3_promos_per_step_log.png")
    print("  exp3_demos_per_step_log.png")
    print("  exp3_promotions_per_episode.png")
    print("  exp3_demotions_per_episode.png")

if __name__ == "__main__":
    main()
