"""Training loop shared by the tabular Q-learning and Double DQN agents."""
import csv
import os
import time
from collections import deque

import numpy as np

from agents.double_dqn_agent import DoubleDQNAgent
from agents.q_learning_agent import QLearningAgent
from game.environment import SnakeEnv

ENV_PARAMS = {"size", "reward_food", "reward_death", "reward_step"}


def make_agent(kind, state_mode="rich", seed=0, **hp):
    # hp goes straight through, so a setting the agent does not have raises TypeError instead of being ignored
    if kind == "q":
        return QLearningAgent(state_mode=state_mode, seed=seed, **hp)
    if kind == "ddqn":
        return DoubleDQNAgent(state_mode=state_mode, seed=seed, **hp)
    raise ValueError(f"unknown agent '{kind}' (use 'q' or 'ddqn')")


def train(kind, episodes=2000, seed=0, state_mode="rich", env_kwargs=None, log_path=None,
          save_path=None, checkpoint_every=None, verbose=True, **hp):
    """Train an agent. Returns (agent, list_of_episode_rows).

    env_kwargs: extra SnakeEnv settings (size, reward_step, ...). Training games use their
    own RNG stream; evaluation uses fixed, separate seeds (training/evaluate.py), so
    training and testing never share games.
    """
    env = SnakeEnv(state_mode=state_mode, seed=seed, **(env_kwargs or {}))
    agent = make_agent(kind, state_mode=state_mode, seed=seed, **hp)
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
            print(f"[{kind}/{state_mode}] episode {ep:>6}/{episodes}  avg score (last 100): "
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
