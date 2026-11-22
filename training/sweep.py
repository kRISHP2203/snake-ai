"""One-at-a-time sweep (Experiments 2 and the reward test).

Changes ONE setting across several values while everything else stays at the default,
repeating each value over several training seeds. Every trained agent is then evaluated
on the same fixed games. Settings can be agent hyperparameters (gamma, lr, ...) or
environment settings (reward_step, ...).
"""
import os

import pandas as pd

from training.evaluate import run_games
from training.train import ENV_PARAMS, train


def sweep(kind, param, values, seeds=(0, 1, 2), episodes=2000, eval_games=100,
          state_mode="rich", out_csv="results/experiment_results.csv", **fixed):
    rows = []
    for value in values:
        for seed in seeds:
            hp, env_kwargs = dict(fixed), {}
            (env_kwargs if param in ENV_PARAMS else hp)[param] = value
            agent, log = train(kind, episodes=episodes, seed=seed, state_mode=state_mode,
                               env_kwargs=env_kwargs, verbose=False, **hp)
            games = run_games(agent, n_games=eval_games, size=env_kwargs.get("size", 10), label=kind)
            mean_score = sum(g["score"] for g in games) / len(games)
            rows.append({"agent": kind, "state_mode": state_mode, "param": param, "value": value,
                         "seed": seed, "episodes": episodes, "eval_mean_score": mean_score,
                         "eval_max_score": max(g["score"] for g in games),
                         "eval_mean_steps": sum(g["steps"] for g in games) / len(games),
                         "train_time_s": log[-1]["elapsed_s"]})
            print(f"[{kind}] {param}={value} seed={seed}: eval mean score {mean_score:.2f}", flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df.to_csv(out_csv, mode="a", header=not os.path.exists(out_csv), index=False)
    return df
