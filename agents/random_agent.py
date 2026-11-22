import random


class RandomAgent:
    """Floor baseline: picks a uniformly random action every step."""
    name = "random"
    state_mode = "basic"      # ignores the state, so any mode works

    def __init__(self, n_actions=3, seed=0):
        self.n_actions = n_actions
        self.rng = random.Random(seed)

    def act(self, state, explore=False):
        return self.rng.randrange(self.n_actions)
