"""Hand-written heuristic baseline (no learning).

Uses only the 11-feature state: prefer a safe move that heads toward the food,
otherwise any safe move, otherwise go straight (it is doomed anyway).
It has no look-ahead, so it can trap itself once the snake gets long.
"""
import random

OFFSETS = (0, 1, -1)   # straight, right, left (matches SnakeEnv actions)


class RuleBasedAgent:
    name = "rule_based"

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
