"""Graphs for the report/presentation, drawn from the CSV files the scripts write.

Colour follows the entity: each agent/state configuration keeps the same colour in every
chart. The four learner colours are the first four slots of a palette checked for colour-
blind separation; hand-written agents are neutral greys. Where a colour is light (aqua,
yellow) the value is also printed as text, so nothing depends on colour alone.
"""
import glob
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
COLORS = {
    "q_basic": "#2a78d6", "q_rich": "#eb6834", "ddqn_basic": "#1baf7a", "ddqn_rich": "#eda100",
    "random": "#c4c3bd", "rule_v1": "#8a8984", "rule_v2": "#52514e",
}
NAMES = {
    "q_basic": "Q-learning, basic state", "q_rich": "Q-learning, rich state",
    "ddqn_basic": "Double DQN, basic state", "ddqn_rich": "Double DQN, rich state",
    "random": "Random", "rule_v1": "Rule-based v1", "rule_v2": "Rule-based v2",
}
# old file/label names from the first version of the project
ALIASES = {"q_learning": "q_basic", "rule_based": "rule_v1"}


def _key(label):
    return ALIASES.get(label, label)


def _color(label):
    return COLORS.get(_key(label), INK2)


def _name(label):
    return NAMES.get(_key(label), label)


def _new_figure(*args, **kwargs):
    fig, axes = plt.subplots(*args, **kwargs)
    fig.patch.set_facecolor(SURFACE)
    for ax in np.atleast_1d(axes).ravel():
        _style(ax)
    return fig, axes


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=3)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)
    ax.title.set_color(INK)


def _save(fig, out_path):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def _legend(ax, **kwargs):
    leg = ax.legend(frameon=False, fontsize=9, **kwargs)
    for text in leg.get_texts():
        text.set_color(INK2)
    return leg


# ---------------------------------------------------------------- training curves
def load_curves(results_dir="results"):
    """Per-episode scores/rewards for every run, in long format (config, seed, episode, ...).

    Uses training_curves.csv from the multi-seed run when present, otherwise falls back to
    the single-run logs written by `main.py train` (training_log_<agent>_<state>.csv).
    """
    path = f"{results_dir}/training_curves.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    frames = []
    for f in sorted(glob.glob(f"{results_dir}/training_log_*.csv")):
        config = os.path.basename(f)[len("training_log_"):-len(".csv")]
        df = pd.read_csv(f)[["episode", "score", "reward"]]
        df.insert(0, "seed", 0)
        df.insert(0, "config", config)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def plot_training_curves(curves, out_path, column="score", window=100):
    """Moving average per seed; line = mean over seeds, band = min to max over seeds."""
    panels = [(t, [c for c in cfgs if c in set(curves["config"])])
              for t, cfgs in (("Q-learning", ["q_basic", "q_rich"]), ("Double DQN", ["ddqn_basic", "ddqn_rich"]))]
    panels = [(t, c) for t, c in panels if c]
    if not panels:
        return
    fig, axes = _new_figure(1, len(panels), figsize=(5.6 * len(panels), 4.2), sharey=True, squeeze=False)
    for ax, (title, configs) in zip(axes[0], panels):
        ends = []
        for config in configs:
            sub = curves[curves.config == config]
            smooth = (sub.pivot(index="episode", columns="seed", values=column)
                      .rolling(window, min_periods=max(1, window // 5)).mean())
            mean, lo, hi = smooth.mean(axis=1), smooth.min(axis=1), smooth.max(axis=1)
            ax.fill_between(smooth.index, lo, hi, color=_color(config), alpha=0.12, linewidth=0)
            ax.plot(smooth.index, mean, color=_color(config), linewidth=2, label=_name(config),
                    solid_capstyle="round")
            ax.plot(smooth.index[-1], mean.iloc[-1], "o", ms=6, color=_color(config),
                    markeredgecolor=SURFACE, markeredgewidth=2)
            ends.append((mean.iloc[-1], smooth.index[-1]))
        lo_y, hi_y = ax.get_ylim()
        gap, last_y = (hi_y - lo_y) * 0.06, None
        for value, x_end in sorted(ends):                 # spread end labels so they never overlap
            label_y = value if last_y is None else max(value, last_y + gap)
            ax.annotate(f"{value:.1f}", (x_end, label_y), xytext=(8, 0), textcoords="offset points",
                        va="center", fontsize=9, color=INK2)
            last_y = label_y
        n_seeds = curves[curves.config.isin(configs)]["seed"].nunique()
        ax.set_title(f"{title}  ({n_seeds} seed{'s' if n_seeds != 1 else ''})", loc="left", fontsize=11)
        ax.set_xlabel("Training episode")
        ax.grid(axis="y", color=GRID, linewidth=1)
        ax.margins(x=0.08)
        _legend(ax, loc="upper left")
    axes[0][0].set_ylabel(f"{column.capitalize()} ({window}-episode moving average)")
    fig.text(0.01, -0.07, "Line = mean over training seeds, shaded band = best and worst seed.",
             fontsize=8.5, color=INK2)
    _save(fig, out_path)


# ---------------------------------------------------------------- multi-seed comparison
def plot_multiseed(csv_path, out_path):
    """Mean over training seeds with the spread between seeds; dots are the individual seeds."""
    df = pd.read_csv(csv_path)
    stats = df.groupby("config")["eval_mean_score"].agg(["mean", "std"]).sort_values("mean")
    fig, ax = _new_figure(figsize=(8.6, 0.62 * len(stats) + 1.8))
    for y, (config, row) in enumerate(stats.iterrows()):
        ax.barh(y, row["mean"], height=0.5, color=_color(config), linewidth=0)
        seeds = df[df.config == config]["eval_mean_score"]
        ax.plot(seeds, np.full(len(seeds), y), "o", ms=5, color=INK, markeredgecolor=SURFACE,
                markeredgewidth=1.5, linestyle="none")
        spread = 0.0 if math.isnan(row["std"]) else row["std"]
        ax.annotate(f"{row['mean']:.1f}  (± {spread:.1f})", (1.01, y), xycoords=("axes fraction", "data"),
                    va="center", fontsize=9, color=INK)
    ax.set_yticks(range(len(stats)))
    ax.set_yticklabels([_name(c) for c in stats.index], color=INK)
    ax.set_xlabel("Mean score on 100 fixed unseen games")
    ax.grid(axis="x", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.set_xlim(0, max(stats["mean"].max(), df["eval_mean_score"].max()) * 1.08)
    n_seeds = df["seed"].nunique()
    ax.set_title(f"Average score by agent, {n_seeds} training seeds each", loc="left", fontsize=11)
    fig.text(0.01, -0.06, "Bar = mean over seeds, dots = individual seeds, ± = std dev between seeds. "
             "Greys are hand-written agents.", fontsize=8.5, color=INK2)
    _save(fig, out_path)


# ---------------------------------------------------------------- single evaluation run
def plot_comparison(eval_csv, out_path):
    """Per-game spread from one evaluation run: mean ± std bar chart and a box plot."""
    df = pd.read_csv(eval_csv)
    df["agent"] = df["agent"].map(_key)
    stats = df.groupby("agent")["score"].agg(["mean", "std"]).sort_values("mean")
    labels = list(stats.index)
    fig, axes = _new_figure(1, 2, figsize=(11.5, 0.5 * len(labels) + 2.6), sharey=True)
    ax = axes[0]
    for y, label in enumerate(labels):
        ax.barh(y, stats.loc[label, "mean"], height=0.5, color=_color(label), linewidth=0)
        ax.annotate(f"{stats.loc[label, 'mean']:.1f}  (± {stats.loc[label, 'std']:.1f})", (1.01, y),
                    xycoords=("axes fraction", "data"), va="center", fontsize=9, color=INK)
    ax.errorbar(stats["mean"], range(len(labels)), xerr=stats["std"], fmt="none", ecolor=INK,
                elinewidth=1, capsize=3)
    ax.set_xlabel("Mean score per game (whisker = std dev between games)")
    ax.set_title("Average score on one evaluation run", loc="left", fontsize=11)
    ax.grid(axis="x", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    box = axes[1]
    parts = box.boxplot([df[df.agent == l]["score"] for l in labels], vert=False, widths=0.5,
                        positions=list(range(len(labels))),
                        patch_artist=True, showfliers=True,
                        medianprops=dict(color=INK, linewidth=1.5),
                        whiskerprops=dict(color=INK2), capprops=dict(color=INK2),
                        flierprops=dict(marker="o", markersize=4, markerfacecolor=INK2, markeredgecolor=SURFACE))
    for patch, label in zip(parts["boxes"], labels):
        patch.set_facecolor(_color(label)); patch.set_edgecolor(INK2); patch.set_linewidth(1)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels([_name(l) for l in labels], color=INK)
    ax.set_ylim(-0.6, len(labels) - 0.4)
    box.set_xlabel("Score per game")
    box.set_title("Spread between games", loc="left", fontsize=11)
    box.grid(axis="x", color=GRID, linewidth=1)
    box.set_axisbelow(True)
    _save(fig, out_path)


# ---------------------------------------------------------------- sweeps
def plot_sweeps_overview(csv_path, out_path):
    """One small panel per (agent, setting): mean evaluation score at each value tried."""
    df = pd.read_csv(csv_path)
    combos = list(dict.fromkeys(zip(df["agent"], df["param"])))
    cols = min(3, len(combos))
    rows = math.ceil(len(combos) / cols)
    fig, axes = _new_figure(rows, cols, figsize=(4.4 * cols, 3.4 * rows), squeeze=False)
    flat = axes.ravel()
    for ax, (agent, param) in zip(flat, combos):
        sub = df[(df.agent == agent) & (df.param == param)]
        color = _color(f"{agent}_{sub['state_mode'].iloc[0] if 'state_mode' in sub else 'basic'}")
        values = sorted(sub["value"].unique())
        pos = {v: i for i, v in enumerate(values)}
        stats = sub.groupby("value")["eval_mean_score"].agg(["mean", "std"]).loc[values]
        ax.errorbar(range(len(values)), stats["mean"], yerr=stats["std"].fillna(0), color=color,
                    linestyle="none", marker="o", ms=9, markeredgecolor=SURFACE, markeredgewidth=2,
                    ecolor=INK2, elinewidth=1, capsize=3, zorder=2)
        ax.plot([pos[v] for v in sub["value"]], sub["eval_mean_score"], "o", ms=3.5, color=INK,
                linestyle="none", zorder=3)
        for i, m in enumerate(stats["mean"]):
            ax.annotate(f"{m:.1f}", (i, m), xytext=(10, 0), textcoords="offset points", ha="left",
                        va="center", fontsize=9, color=INK)
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels([f"{v:g}" for v in values])
        ax.set_xlim(-0.4, len(values) - 0.35)
        ax.set_ylim(0, max(stats["mean"] + stats["std"].fillna(0)) * 1.25 + 0.5)
        ax.set_xlabel(param)
        ax.set_title(f"{'Double DQN' if agent == 'ddqn' else 'Q-learning'}: {param}", loc="left", fontsize=11)
        ax.grid(axis="y", color=GRID, linewidth=1)
        ax.set_axisbelow(True)
    for ax in flat[len(combos):]:
        ax.set_visible(False)
    axes[0][0].set_ylabel("Eval mean score")
    for r in range(1, rows):
        axes[r][0].set_ylabel("Eval mean score")
    fig.text(0.01, -0.01, "Colour dot = mean over training seeds, whisker = std dev between seeds, "
             "black dots = individual seeds. Each panel changes one setting only; y axes differ per panel.",
             fontsize=8.5, color=INK2)
    fig.tight_layout()
    _save(fig, out_path)


# ---------------------------------------------------------------- grid-size generalisation
def plot_generalisation(csv_path, out_path):
    """Models trained on 10x10 and evaluated, without retraining, on other grid sizes."""
    df = pd.read_csv(csv_path)
    df = df[df.agent != "random"]            # always about 0, only compresses the axis
    fig, ax = _new_figure(figsize=(7.4, 4.4))
    order = ["rule_v1", "rule_v2", "q_basic", "q_rich", "ddqn_basic", "ddqn_rich"]
    for agent in [a for a in order if a in set(df.agent)]:
        sub = df[df.agent == agent].sort_values("grid_size")
        ax.plot(sub["grid_size"], sub["mean_score"], color=_color(agent), linewidth=2, marker="o", ms=6,
                markeredgecolor=SURFACE, markeredgewidth=2, label=_name(agent))
    sizes = sorted(df["grid_size"].unique())
    ax.set_xticks(sizes)
    ax.set_xticklabels([f"{s}x{s}" for s in sizes])
    ax.set_xlabel("Grid size (trained on 10x10 only)")
    ax.set_ylabel("Mean score on 100 fixed games")
    ax.set_title("Does the learned behaviour carry over to other grid sizes?", loc="left", fontsize=11)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_ylim(bottom=0)
    _legend(ax, loc="upper left", ncols=2)
    fig.text(0.01, -0.08, "Bigger grids give higher scores for every agent, so compare the order of the lines, "
             "not the raw numbers. Random (about 0) is omitted.", fontsize=8.5, color=INK2)
    _save(fig, out_path)
