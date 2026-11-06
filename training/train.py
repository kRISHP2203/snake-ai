"""Training loop shared by the tabular Q-learning and DQN agents."""
import csv
import os
import time
from collections import deque

import numpy as np

from agents.dqn_agent import DQNAgent
from agents.q_learning_agent import QLearningAgent
from game.environment import SnakeEnv


def make_agent(kind, seed=0, **hp):
    if kind == "q":
        allowed = {"alpha", "gamma", "eps_decay", "eps_end"}
        return QLearningAgent(seed=seed, **{k: v for k, v in hp.items() if k in allowed})
    if kind == "dqn":
        allowed = {"lr", "gamma", "eps_decay", "eps_end", "batch_size", "hidden", "target_update"}
        return DQNAgent(seed=seed, **{k: v for k, v in hp.items() if k in allowed})
    raise ValueError(f"unknown agent '{kind}' (use 'q' or 'dqn')")


def train(kind, episodes=2000, seed=0, size=10, log_path=None, save_path=None,
          checkpoint_every=None, verbose=True, **hp):
    """Train an agent. Returns (agent, list_of_episode_rows).

    Training games use their own RNG stream; evaluation uses fixed, separate seeds
    (see training/evaluate.py), so training and testing never share games.
    """
    env = SnakeEnv(size=size, seed=seed)
    agent = make_agent(kind, seed=seed, **hp)
    rows, recent = [], deque(maxlen=100)
    started = time.time()

    for ep in range(1, episodes + 1):
        state = env.reset()
        done, total_reward = False, 0.0
        while not done:
            action = agent.act(state, explore=True)
            next_state, reward, done, info = env.step(action)
            agent.learn(state, action, reward, next_state, done)
            state, total_reward = next_state, total_reward + reward
        agent.end_episode()
        recent.append(env.score)
        rows.append({"episode": ep, "score": env.score, "reward": total_reward,
                     "steps": env.steps, "epsilon": agent.epsilon, "cause": info["cause"],
                     "elapsed_s": time.time() - started})

        if verbose and (ep % max(1, episodes // 10) == 0 or ep == episodes):
            print(f"[{kind}] episode {ep:>6}/{episodes}  avg score (last 100): "
                  f"{np.mean(recent):5.2f}  best: {max(r['score'] for r in rows):>3}  "
                  f"epsilon: {agent.epsilon:.3f}")
        if save_path and checkpoint_every and ep % checkpoint_every == 0:
            agent.save(f"{save_path}_ep{ep}.npz")

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        agent.save(f"{save_path}.npz")
    if log_path:
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        with open(log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    return agent, rows
