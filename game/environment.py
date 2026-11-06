"""Headless, seedable Snake environment with a Gym-style reset()/step() API.

Actions are RELATIVE to the snake's current heading:
    0 = go straight, 1 = turn right, 2 = turn left

State is an 11-element binary vector:
    [danger_straight, danger_right, danger_left,
     heading_up, heading_right, heading_down, heading_left,
     food_up, food_right, food_down, food_left]
"""
from __future__ import annotations

import random

import numpy as np

UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3          # clockwise: right turn = +1, left = -1
DELTAS = {UP: (0, -1), RIGHT: (1, 0), DOWN: (0, 1), LEFT: (-1, 0)}
STRAIGHT, TURN_RIGHT, TURN_LEFT = 0, 1, 2
TURN_OFFSET = {STRAIGHT: 0, TURN_RIGHT: 1, TURN_LEFT: -1}


class SnakeEnv:
    n_actions = 3
    state_size = 11

    def __init__(self, size=10, reward_food=10.0, reward_death=-10.0,
                 reward_step=0.0, idle_limit_factor=100, seed=None):
        if size < 5:
            raise ValueError("size must be at least 5")
        self.size = size
        self.reward_food = reward_food
        self.reward_death = reward_death
        self.reward_step = reward_step
        self.idle_limit_factor = idle_limit_factor   # max steps without food = factor * length
        self.rng = random.Random(seed)
        self.reset()

    # ------------------------------------------------------------------ API
    def reset(self, seed=None):
        """Start a new game. Pass a seed to make the food sequence reproducible."""
        if seed is not None:
            self.rng = random.Random(seed)
        mid = self.size // 2
        self.direction = RIGHT
        self.snake = [(mid, mid), (mid - 1, mid), (mid - 2, mid)]   # head first
        self.score = 0
        self.steps = 0
        self.steps_since_food = 0
        self.done = False
        self.food = None
        self.place_food()
        return self.get_state()

    def step(self, action):
        """Apply an action. Returns (next_state, reward, done, info)."""
        if self.done:
            raise RuntimeError("Game is over: call reset() first.")
        self.direction = (self.direction + TURN_OFFSET[int(action)]) % 4
        dx, dy = DELTAS[self.direction]
        head = (self.snake[0][0] + dx, self.snake[0][1] + dy)
        self.steps += 1
        self.steps_since_food += 1

        ate = head == self.food
        body = self.snake if ate else self.snake[:-1]   # tail moves away unless we grow
        info = {"cause": None, "won": False}

        if self.check_collision(head, body):
            self.done = True
            reward = self.reward_death
            info["cause"] = "wall" if self._outside(head) else "self"
        else:
            self.snake.insert(0, head)
            if ate:
                self.score += 1
                self.steps_since_food = 0
                reward = self.reward_food
                self.place_food()
                if self.food is None:                    # board full = perfect game
                    self.done = True
                    info["won"] = True
                    info["cause"] = "won"
            else:
                self.snake.pop()
                reward = self.reward_step
            if not self.done and self.steps_since_food > self.idle_limit_factor * len(self.snake):
                self.done = True
                reward = self.reward_death
                info["cause"] = "timeout"

        info["score"] = self.score
        info["steps"] = self.steps
        return self.get_state(), reward, self.done, info

    def get_state(self):
        head = self.snake[0]
        d = self.direction
        body = self.snake[:-1]                           # tail will move, so it is not an obstacle
        danger = [self._blocked_in_direction(head, (d + off) % 4, body) for off in (0, 1, -1)]
        heading = [d == UP, d == RIGHT, d == DOWN, d == LEFT]
        if self.food is None:
            food = [False] * 4
        else:
            fx, fy = self.food
            hx, hy = head
            food = [fy < hy, fx > hx, fy > hy, fx < hx]
        return np.array(danger + heading + food, dtype=np.float32)

    # ------------------------------------------------------------ mechanics
    def place_food(self):
        occupied = set(self.snake)
        free = [(x, y) for x in range(self.size) for y in range(self.size) if (x, y) not in occupied]
        self.food = self.rng.choice(free) if free else None

    def check_collision(self, pt, body=None):
        body = self.snake if body is None else body
        return self._outside(pt) or pt in body

    def _outside(self, pt):
        return not (0 <= pt[0] < self.size and 0 <= pt[1] < self.size)

    def _blocked_in_direction(self, head, direction, body):
        dx, dy = DELTAS[direction]
        return self.check_collision((head[0] + dx, head[1] + dy), body)

    # ------------------------------------------------------------ rendering
    def render_ascii(self):
        rows = []
        for y in range(self.size):
            row = ""
            for x in range(self.size):
                if (x, y) == self.snake[0]:
                    row += "O "
                elif (x, y) in self.snake:
                    row += "o "
                elif (x, y) == self.food:
                    row += "* "
                else:
                    row += ". "
            rows.append(row)
        return "\n".join(rows) + f"\nscore={self.score} steps={self.steps}"
