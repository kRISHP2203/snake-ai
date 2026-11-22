"""Evaluation: fixed seeds, no exploration, no learning. Same games for every agent."""
import os

import pandas as pd

from game.environment import SnakeEnv

EVAL_SEED_START = 1_000_000   # far from any training seed stream


def run_games(agent, n_games=100, size=10, seed_start=EVAL_SEED_START, label=None):
    """Play n_games with fixed seeds. The env uses the agent's own state_mode, but the
    food sequence depends only on the seed, so every agent faces the same games."""
    env = SnakeEnv(size=size, state_mode=getattr(agent, "state_mode", "basic"))
    rows = []
    for g in range(n_games):
        state = env.reset(seed=seed_start + g)
        done, total_reward = False, 0.0
        while not done:
            state, reward, done, info = env.step(agent.act(state, explore=False))
            total_reward += reward
        rows.append({"agent": label or agent.name, "game": g, "score": env.score,
                     "steps": env.steps, "reward": total_reward, "cause": info["cause"]})
    return rows


def summarise(df):
    g = df.groupby("agent")
    out = g.agg(mean_score=("score", "mean"), std_score=("score", "std"),
                max_score=("score", "max"), mean_steps=("steps", "mean"),
                mean_reward=("reward", "mean"), games=("score", "size"))
    return out.round(2).sort_values("mean_score", ascending=False)


def evaluate_agents(agents, n_games=100, size=10, out_csv=None):
    """agents: dict of label -> agent object. Returns (per-game DataFrame, summary DataFrame)."""
    rows = []
    for label, agent in agents.items():
        rows += run_games(agent, n_games, size, label=label)
    df = pd.DataFrame(rows)
    if out_csv:
        os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
        df.to_csv(out_csv, index=False)
    return df, summarise(df)
