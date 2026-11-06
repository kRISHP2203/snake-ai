"""One-at-a-time hyperparameter sweep (Experiment 2).

Changes ONE parameter across several values while everything else stays at the
default, repeating each setting over several training seeds. Each trained agent is
then evaluated on the same fixed games.
"""
import os

import pandas as pd

from training.evaluate import run_games
from training.train import train


def sweep(kind, param, values, seeds=(0, 1, 2), episodes=2000, eval_games=100,
          out_csv="results/experiment_results.csv", **fixed):
    rows = []
    for value in values:
        for seed in seeds:
            hp = dict(fixed, **{param: value})
            agent, log = train(kind, episodes=episodes, seed=seed, verbose=False, **hp)
            games = run_games(agent, n_games=eval_games, label=f"{kind}")
            mean_score = sum(g["score"] for g in games) / len(games)
            rows.append({"agent": kind, "param": param, "value": value, "seed": seed,
                         "episodes": episodes, "eval_mean_score": mean_score,
                         "eval_max_score": max(g["score"] for g in games),
                         "eval_mean_steps": sum(g["steps"] for g in games) / len(games),
                         "train_time_s": log[-1]["elapsed_s"]})
            print(f"{param}={value} seed={seed}: eval mean score {mean_score:.2f}")
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    header = not os.path.exists(out_csv)
    df.to_csv(out_csv, mode="a", header=header, index=False)
    return df
