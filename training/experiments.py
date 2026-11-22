"""Multi-seed comparison (Experiment 1) and grid-size generalisation test."""
import csv
import os
import time

import pandas as pd

from agents.random_agent import RandomAgent
from agents.rule_based_agent import RuleBasedAgent, RuleBasedV2Agent
from training import config
from training.evaluate import run_games
from training.train import train

CONFIGS = [("q", "basic"), ("q", "rich"), ("ddqn", "basic"), ("ddqn", "rich")]


def run_multiseed(seeds=config.SEEDS, eval_games=config.EVAL_GAMES, episodes=None,
                  out_dir="results", model_dir="results/models", configs=CONFIGS):
    """Train every (agent, state) config over several seeds and evaluate on the same games.

    Writes multiseed_results.csv (one row per trained agent or baseline run) and
    training_curves.csv (score per episode for every run). Seed 0 of each config is saved
    as results/models/<agent>_<state>.npz. Results are flushed after every run.
    """
    episodes = episodes or config.EPISODES
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    res_path, curve_path = f"{out_dir}/multiseed_results.csv", f"{out_dir}/training_curves.csv"
    res_fields = ["config", "agent", "state_mode", "seed", "episodes", "eval_mean_score",
                  "eval_std_score", "eval_max_score", "eval_mean_steps", "train_time_s"]
    with open(res_path, "w", newline="") as rf, open(curve_path, "w", newline="") as cf:
        res = csv.DictWriter(rf, fieldnames=res_fields); res.writeheader()
        cur = csv.writer(cf); cur.writerow(["config", "seed", "episode", "score", "reward"])

        def record(label, kind, mode, seed, eps, games, train_time):
            scores = pd.Series([g["score"] for g in games])
            res.writerow({"config": label, "agent": kind, "state_mode": mode, "seed": seed,
                          "episodes": eps, "eval_mean_score": scores.mean(),
                          "eval_std_score": scores.std(), "eval_max_score": scores.max(),
                          "eval_mean_steps": sum(g["steps"] for g in games) / len(games),
                          "train_time_s": train_time})
            rf.flush()
            print(f"{label:<10} seed {seed}: eval mean score {scores.mean():.2f}", flush=True)

        for seed in range(seeds):                       # baselines: only their tie-breaking is random
            for label, agent, mode in (("random", RandomAgent(seed=seed), "basic"),
                                       ("rule_v1", RuleBasedAgent(seed=seed), "basic"),
                                       ("rule_v2", RuleBasedV2Agent(seed=seed), "rich")):
                record(label, label, mode, seed, 0, run_games(agent, eval_games), 0.0)

        for kind, mode in configs:
            label = f"{kind}_{mode}"
            for seed in range(seeds):
                save = f"{model_dir}/{label}" if seed == 0 else None
                agent, log = train(kind, episodes=episodes[kind], seed=seed, state_mode=mode,
                                   save_path=save, verbose=False)
                for r in log:
                    cur.writerow([label, seed, r["episode"], r["score"], r["reward"]])
                cf.flush()
                record(label, kind, mode, seed, episodes[kind], run_games(agent, eval_games),
                       log[-1]["elapsed_s"])
    return pd.read_csv(res_path)


def summarise_multiseed(df):
    """Mean and spread ACROSS training seeds (each seed's score is already a 100-game mean)."""
    g = df.groupby("config")
    out = g.agg(mean_score=("eval_mean_score", "mean"), std_across_seeds=("eval_mean_score", "std"),
                worst_seed=("eval_mean_score", "min"), best_seed=("eval_mean_score", "max"),
                mean_steps=("eval_mean_steps", "mean"), train_time_s=("train_time_s", "mean"),
                seeds=("seed", "nunique"))
    return out.round(2).sort_values("mean_score", ascending=False)


def run_generalisation(model_dir="results/models", sizes=config.GRID_SIZES,
                       eval_games=config.EVAL_GAMES, out_csv="results/generalisation_results.csv"):
    """Evaluate models trained on 10x10 (and the baselines) on other grid sizes, no retraining."""
    from agents.double_dqn_agent import DoubleDQNAgent
    from agents.q_learning_agent import QLearningAgent

    def build_agents():
        """Fresh agents every time, so each result is independent of the order they are run in."""
        agents = {"random": RandomAgent(), "rule_v1": RuleBasedAgent(), "rule_v2": RuleBasedV2Agent()}
        for kind, mode in CONFIGS:
            path = f"{model_dir}/{kind}_{mode}.npz"
            if os.path.exists(path):
                agents[f"{kind}_{mode}"] = (QLearningAgent() if kind == "q" else DoubleDQNAgent()).load(path)
        return agents

    rows = []
    for size in sizes:
        for label, agent in build_agents().items():
            games = pd.DataFrame(run_games(agent, eval_games, size=size, label=label))
            rows.append({"agent": label, "grid_size": size, "mean_score": games.score.mean(),
                         "std_score": games.score.std(), "max_score": games.score.max(),
                         "mean_steps": games.steps.mean(), "games": len(games)})
            print(f"size {size:>2} {label:<10} mean score {games.score.mean():5.2f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return df
