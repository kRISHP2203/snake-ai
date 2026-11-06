# Snake AI

A small, self-contained Snake reinforcement-learning project in Python.
Everything runs headless on a CPU with only **NumPy, Pandas and Matplotlib** (Pygame is optional, for the viewer).

## What's in it

| Agent | Type | Learns? |
|---|---|---|
| `random` | picks a random action | no |
| `rule_based` | heads for food, avoids immediate collisions | no |
| `q_learning` | tabular Q-learning (2048 states x 3 actions) | yes |
| `dqn` | Deep Q-Network written in NumPy (MLP, replay buffer, target network, Adam) | yes |

**Environment:** 10x10 grid, Gym-style `reset()` / `step(action)`, seedable, step-limit to stop endless loops.
**Actions (relative):** straight, turn right, turn left.
**State (11 binary features):** danger straight/right/left, heading (4), food up/right/down/left.
**Reward:** +10 for food, -10 for dying (wall, self, or timeout). Edit in `game/environment.py`.

## Quick start

```bash
pip install -r requirements.txt

python main.py train --agent q   --episodes 3000     # ~7 s
python main.py train --agent dqn --episodes 1500     # ~1 min
python main.py evaluate --games 100                  # compares all agents on fixed unseen games
python main.py plot                                  # graphs -> results/graphs/
python main.py watch --agent dqn                     # Pygame window (or add --ascii for text mode)
```

Pre-trained models from my test run are in `results/models/`, so `watch` works straight away.

## Change hyperparameters / run experiments

```bash
python main.py train --agent dqn --lr 0.0005 --gamma 0.95 --eps-decay 0.99 --batch-size 128
python main.py sweep --agent q --param gamma --values 0.5 0.9 0.99 --seeds 3 --episodes 2000
python main.py plot
```

`sweep` changes ONE parameter at a time, repeats over several training seeds, and evaluates each trained agent on the same fixed games. Results append to `results/experiment_results.csv`.

## Layout

```
game/environment.py     rules, state, reward           game/render.py   optional Pygame viewer
agents/                 random, rule_based, q_learning, dqn
training/train.py       training loop + logging        training/evaluate.py   fixed-seed evaluation
training/sweep.py       one-at-a-time parameter sweep  analysis/plots.py      graphs
main.py                 command-line interface         results/               logs, models, graphs
```

Training and evaluation are separate: evaluation uses seeds starting at 1,000,000 with exploration off and no learning, so agents are never tested on games they trained on.

## Results from my test run (single training seed, 10x10 grid, 100 evaluation games)

| agent | mean score | std | max |
|---|---|---|---|
| dqn (1500 episodes) | 18.44 | 5.73 | 32 |
| rule_based | 17.30 | 6.18 | 36 |
| q_learning (3000 episodes) | 15.88 | 5.55 | 29 |
| random | 0.11 | 0.31 | 1 |

Honest reading: both learners are far better than random, but the differences between DQN, Q-learning and the rule-based agent are small relative to the game-to-game spread, and each learner was trained from only **one seed**. Don't claim a ranking until you've repeated training over several seeds (`sweep` does this).

## Known limitations / ideas to improve

* The 11-feature state only sees one step ahead, so agents still trap themselves once the snake is long. Adding features (distance to walls, free space reachable via flood-fill) is the most effective upgrade.
* Rewards are sparse. Try `reward_step` (small negative) or a distance-to-food shaping term, and compare.
* The DQN is plain (no double-DQN, no prioritized replay). It is easy to add double-DQN in `DQNAgent.learn`.
* The Pygame viewer (`game/render.py`) compiles but I could not run it in my sandbox (no display or Pygame there). The `--ascii` mode was tested.
