"""Shared defaults so experiments are reproducible and easy to change in one place."""

SIZE = 10                  # grid used for training and the main comparison
EVAL_GAMES = 100           # games per evaluation (fixed seeds, no learning, no exploration)
SEEDS = 5                  # training seeds per configuration in the multi-seed comparison
EPISODES = {"q": 3000, "ddqn": 1500}
SWEEP_EPISODES = {"q": 2000, "ddqn": 1000}
SWEEP_SEEDS = 3

# One-at-a-time sweeps: (agent, parameter, values). The default for each parameter is
# the value the agent uses when you do not override it (see agents/*.py).
SWEEPS = [
    ("q",   "gamma",     [0.5, 0.9, 0.99]),
    ("q",   "alpha",     [0.05, 0.1, 0.3]),
    ("q",   "eps_decay", [0.99, 0.998, 0.9995]),
    ("ddqn", "lr",        [0.0003, 0.001, 0.003]),
    ("ddqn", "gamma",     [0.5, 0.9, 0.99]),
    ("ddqn", "eps_decay", [0.98, 0.995, 0.999]),
    ("ddqn", "batch_size", [16, 64, 256]),
    # reward experiment: small penalty on every step that does not eat
    ("q",   "reward_step", [0.0, -0.01, -0.1]),
    ("ddqn", "reward_step", [0.0, -0.01, -0.1]),
]
GRID_SIZES = (8, 10, 12, 15)
