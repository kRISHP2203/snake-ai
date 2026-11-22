"""Double Deep Q-Network (Double DQN) in pure NumPy (no PyTorch needed).

Pieces: a small MLP Q-network trained with Adam, an experience replay buffer,
a periodically-synced target network, and epsilon-greedy exploration.
Loss is mean-squared TD error with the gradient clipped to [-1, 1] (Huber-like).

Why "double": plain DQN builds its target as r + gamma * max_a Q_target(s', a). Taking
the max over noisy estimates picks the action whose value happens to be overestimated,
so values drift upward. Double DQN separates the two jobs (van Hasselt, Guez and Silver,
2016): the ONLINE network chooses the best next action, the TARGET network scores it:

    a* = argmax_a Q_online(s', a)
    target = r + gamma * Q_target(s', a*)        (just r if the game ended)
"""
import numpy as np

from game.environment import STATE_SIZES


class QNetwork:
    def __init__(self, in_dim, hidden, out_dim, lr, rng):
        def he(i, o):
            return rng.standard_normal((i, o)) * np.sqrt(2.0 / i)
        self.p = {"W1": he(in_dim, hidden), "b1": np.zeros(hidden),
                  "W2": he(hidden, hidden), "b2": np.zeros(hidden),
                  "W3": he(hidden, out_dim) * 0.1, "b3": np.zeros(out_dim)}
        self.lr = lr
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}   # Adam moments
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.t = 0

    def forward(self, x):
        p = self.p
        z1 = x @ p["W1"] + p["b1"]; a1 = np.maximum(z1, 0)
        z2 = a1 @ p["W2"] + p["b2"]; a2 = np.maximum(z2, 0)
        out = a2 @ p["W3"] + p["b3"]
        return out, (x, z1, a1, z2, a2)

    def predict(self, x):
        return self.forward(x)[0]

    def train_batch(self, x, actions, targets):
        out, (x, z1, a1, z2, a2) = self.forward(x)
        idx = np.arange(len(x))
        td = out[idx, actions] - targets
        g = np.zeros_like(out)
        g[idx, actions] = np.clip(td, -1.0, 1.0) / len(x)           # clipped gradient
        grads = {}
        grads["W3"] = a2.T @ g;            grads["b3"] = g.sum(0)
        d2 = (g @ self.p["W3"].T) * (z2 > 0)
        grads["W2"] = a1.T @ d2;           grads["b2"] = d2.sum(0)
        d1 = (d2 @ self.p["W2"].T) * (z1 > 0)
        grads["W1"] = x.T @ d1;            grads["b1"] = d1.sum(0)
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for k in self.p:                                             # Adam update
            self.m[k] = b1 * self.m[k] + (1 - b1) * grads[k]
            self.v[k] = b2 * self.v[k] + (1 - b2) * grads[k] ** 2
            mh = self.m[k] / (1 - b1 ** self.t)
            vh = self.v[k] / (1 - b2 ** self.t)
            self.p[k] -= self.lr * mh / (np.sqrt(vh) + eps)
        return float(np.mean(td ** 2))

    def copy_from(self, other):
        self.p = {k: v.copy() for k, v in other.p.items()}


class ReplayBuffer:
    def __init__(self, capacity, state_size):
        self.cap, self.n, self.i = capacity, 0, 0
        self.s = np.zeros((capacity, state_size), np.float32)
        self.s2 = np.zeros((capacity, state_size), np.float32)
        self.a = np.zeros(capacity, np.int64)
        self.r = np.zeros(capacity, np.float32)
        self.d = np.zeros(capacity, np.float32)

    def add(self, s, a, r, s2, d):
        i = self.i
        self.s[i], self.a[i], self.r[i], self.s2[i], self.d[i] = s, a, r, s2, d
        self.i = (i + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def sample(self, batch, rng):
        ix = rng.integers(0, self.n, size=batch)
        return self.s[ix], self.a[ix], self.r[ix], self.s2[ix], self.d[ix]


class DoubleDQNAgent:
    name = "ddqn"

    def __init__(self, n_actions=3, state_mode="basic", hidden=64, lr=1e-3, gamma=0.9,
                 batch_size=64, buffer_size=50_000, warmup=1000, train_every=1,
                 target_update=500, eps_start=1.0, eps_end=0.01, eps_decay=0.995, seed=0):
        state_size = STATE_SIZES[state_mode]
        self.n_actions, self.state_size, self.hidden = n_actions, state_size, hidden
        self.state_mode = state_mode
        self.gamma, self.batch_size = gamma, batch_size
        self.warmup, self.train_every, self.target_update = warmup, train_every, target_update
        self.epsilon, self.eps_end, self.eps_decay = eps_start, eps_end, eps_decay
        self.rng = np.random.default_rng(seed)
        self.net = QNetwork(state_size, hidden, n_actions, lr, self.rng)
        self.target = QNetwork(state_size, hidden, n_actions, lr, self.rng)
        self.target.copy_from(self.net)
        self.buffer = ReplayBuffer(buffer_size, state_size)
        self.steps = 0
        self.last_loss = None

    def act(self, state, explore=False):
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        return int(np.argmax(self.net.predict(state[None, :])[0]))

    def q_values(self, state):
        """Handy for decision-visualisation: the network's value for each action."""
        return self.net.predict(state[None, :])[0]

    def learn(self, state, action, reward, next_state, done):
        self.buffer.add(state, action, reward, next_state, float(done))
        self.steps += 1
        if self.buffer.n >= max(self.warmup, self.batch_size) and self.steps % self.train_every == 0:
            s, a, r, s2, d = self.buffer.sample(self.batch_size, self.rng)
            targets = self.double_dqn_targets(r, s2, d)
            self.last_loss = self.net.train_batch(s, a, targets)
        if self.steps % self.target_update == 0:
            self.target.copy_from(self.net)

    def double_dqn_targets(self, rewards, next_states, dones):
        """Online net picks the next action, target net evaluates it."""
        best_next_actions = np.argmax(self.net.predict(next_states), axis=1)
        target_values = self.target.predict(next_states)
        next_q = target_values[np.arange(len(next_states)), best_next_actions]
        return rewards + self.gamma * next_q * (1.0 - dones)

    def end_episode(self):
        self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay)

    def save(self, path):
        np.savez(path, epsilon=self.epsilon, hidden=self.hidden,
                 state_mode=self.state_mode, **self.net.p)

    def load(self, path):
        data = np.load(path)
        mode = str(data["state_mode"]) if "state_mode" in data else "basic"
        self.__init__(n_actions=self.n_actions, state_mode=mode, hidden=int(data["hidden"]))
        self.net.p = {k: data[k] for k in self.net.p}
        self.target.copy_from(self.net)
        self.epsilon = float(data["epsilon"])
        return self
