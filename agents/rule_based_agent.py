"""Hand-written heuristic baselines (no learning).

RuleBasedAgent (v1, FROZEN): uses only the 11 basic features. It prefers a safe move
that heads toward the food, otherwise any safe move. It has no look-ahead, so it traps
itself once the snake gets long. Do not change this class: it is the reference baseline.

RuleBasedV2Agent: same idea plus the 3 flood-fill "trap" flags of the rich state, so it
also avoids moves that lead into a pocket smaller than the snake. It shows how far
hand-written rules can go with the same information the learners get.
"""
import random

OFFSETS = (0, 1, -1)   # straight, right, left (matches SnakeEnv actions)


class RuleBasedAgent:
    name = "rule_v1"
    state_mode = "basic"

    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def act(self, state, explore=False):
        danger = state[0:3]
        heading = int(list(state[3:7]).index(1.0))
        food = state[7:11]                      # up, right, down, left (same order as headings)
        safe = [a for a in range(3) if not danger[a]]
        if not safe:
            return 0
        toward = [a for a in safe if food[(heading + OFFSETS[a]) % 4]]
        pool = toward or safe
        return 0 if 0 in pool else self.rng.choice(pool)


class RuleBasedV2Agent:
    name = "rule_v2"
    state_mode = "rich"

    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def act(self, state, explore=False):
        danger = state[0:3]
        trap = state[11:14]
        heading = int(list(state[3:7]).index(1.0))
        food = state[7:11]
        safe = [a for a in range(3) if not danger[a]]
        if not safe:
            return 0
        open_moves = [a for a in safe if not trap[a]] or safe     # avoid pockets if possible
        toward = [a for a in open_moves if food[(heading + OFFSETS[a]) % 4]]
        pool = toward or open_moves
        return 0 if 0 in pool else self.rng.choice(pool)
