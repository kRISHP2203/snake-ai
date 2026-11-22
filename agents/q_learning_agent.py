"""Tabular Q-learning. The binary features index a (2**n_features) x 3 table directly."""
import numpy as np

from game.environment import STATE_SIZES


class QLearningAgent:
    name = "q_learning"

    def __init__(self, n_actions=3, state_mode="basic", alpha=0.1, gamma=0.9,
                 eps_start=1.0, eps_end=0.01, eps_decay=0.998, seed=0):
        self.n_actions = n_actions
        self.state_mode = state_mode
        state_size = STATE_SIZES[state_mode]
        self.alpha, self.gamma = alpha, gamma
        self.epsilon, self.eps_end, self.eps_decay = eps_start, eps_end, eps_decay
        self.rng = np.random.default_rng(seed)
        self.q = np.zeros((2 ** state_size, n_actions))
        self._powers = 2 ** np.arange(state_size)

    def _key(self, state):
        return int(np.dot(state.astype(np.int64), self._powers))

    def act(self, state, explore=False):
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        values = self.q[self._key(state)]
        return int(self.rng.choice(np.flatnonzero(values == values.max())))   # random tie-break

    def q_values(self, state):
        """The table's value for each action (used by the visual viewers)."""
        return self.q[self._key(state)].copy()

    def learn(self, state, action, reward, next_state, done):
        s, s2 = self._key(state), self._key(next_state)
        target = reward if done else reward + self.gamma * self.q[s2].max()
        self.q[s, action] += self.alpha * (target - self.q[s, action])

    def end_episode(self):
        self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay)

    def save(self, path):
        np.savez(path, q=self.q, epsilon=self.epsilon, state_mode=self.state_mode)

    def load(self, path):
        data = np.load(path)
        mode = str(data["state_mode"]) if "state_mode" in data else "basic"
        self.__init__(n_actions=self.n_actions, state_mode=mode)
        self.q = data["q"]
        self.epsilon = float(data["epsilon"])
        return self
