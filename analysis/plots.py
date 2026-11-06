"""Graphs for the report/presentation, drawn from the CSV files the scripts write."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_training(log_csvs, out_path, window=100, column="score"):
    """log_csvs: dict label -> training_log.csv path. Moving average of score (or reward)."""
    plt.figure(figsize=(8, 4.5))
    for label, path in log_csvs.items():
        df = pd.read_csv(path)
        plt.plot(df["episode"], df[column].rolling(window, min_periods=1).mean(), label=label)
    plt.xlabel("Training episode")
    plt.ylabel(f"{column} (moving average, window={window})")
    plt.title(f"Training progress: {column}")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150); plt.close()


def plot_comparison(eval_csv, out_path):
    """Bar chart of mean score per agent (error bars = std dev) plus a score box plot."""
    df = pd.read_csv(eval_csv)
    stats = df.groupby("agent")["score"].agg(["mean", "std"]).sort_values("mean")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(stats.index, stats["mean"], yerr=stats["std"], capsize=5)
    axes[0].set_ylabel("Mean score per game"); axes[0].set_title("Average score by agent (± std dev)")
    labels = list(stats.index)
    data = [df[df.agent == a]["score"] for a in labels]
    try:
        axes[1].boxplot(data, tick_labels=labels)      # Matplotlib >= 3.9
    except TypeError:
        axes[1].boxplot(data, labels=labels)           # older Matplotlib
    axes[1].set_ylabel("Score"); axes[1].set_title("Score distribution (variability)")
    for ax in axes:
        ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150); plt.close()


def plot_sweep(sweep_csv, param, out_path):
    """Mean evaluation score per parameter value, with spread across training seeds."""
    df = pd.read_csv(sweep_csv)
    df = df[df.param == param]
    stats = df.groupby("value")["eval_mean_score"].agg(["mean", "std"]).reset_index()
    plt.figure(figsize=(6.5, 4.5))
    plt.errorbar(stats["value"].astype(str), stats["mean"], yerr=stats["std"], fmt="o-", capsize=5)
    plt.xlabel(param); plt.ylabel("Evaluation mean score")
    plt.title(f"Effect of {param} (error bars = std dev over seeds)")
    plt.grid(alpha=0.3); plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150); plt.close()
