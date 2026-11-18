"""
Size-independent Deep Q-Network scheduler (our own design, PyTorch).

Instead of one output per server (which ties the model to a fixed cluster size), one shared
network scores every candidate server:

        Q(state, server_i) = MLP( task features  +  server_i features  +  global features )

The scheduler picks the feasible server with the highest Q-value, so the same trained model
works for 5, 10 or 20 servers.

Standard DQN components (Mnih et al., 2015): experience replay, epsilon-greedy exploration,
a periodically synchronised target network, Huber loss, Adam optimiser.
"""
import numpy as np
import torch
import torch.nn as nn

import config
from src.cloud_environment import N_FEATURES

torch.set_num_threads(1)   # small network: one thread is fastest and keeps runs reproducible


class QNetwork(nn.Module):
    def __init__(self, n_features=N_FEATURES, hidden=config.HIDDEN):
        super().__init__()
        layers, d = [], n_features
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):                      # x [..., n_features] -> [...]
        return self.net(x).squeeze(-1)


class ReplayMemory:
    """Stores (chosen-action features, reward, next-state candidate features, mask, done)."""

    def __init__(self, capacity=config.REPLAY_SIZE, max_servers=config.MAX_SERVERS, nf=N_FEATURES):
        self.cap, self.ms = capacity, max_servers
        self.sa = np.zeros((capacity, nf), np.float32)
        self.r = np.zeros(capacity, np.float32)
        self.nx = np.zeros((capacity, max_servers, nf), np.float32)
        self.nm = np.zeros((capacity, max_servers), bool)
        self.done = np.zeros(capacity, np.float32)
        self.i, self.full = 0, False

    def push(self, sa, r, next_feats, next_mask, done):
        k = self.i
        self.sa[k], self.r[k], self.done[k] = sa, r, float(done)
        self.nx[k] = 0
        self.nm[k] = False
        if next_feats is not None:
            n = len(next_feats)
            self.nx[k, :n], self.nm[k, :n] = next_feats, next_mask
        self.i = (k + 1) % self.cap
        self.full = self.full or self.i == 0

    def __len__(self):
        return self.cap if self.full else self.i

    def sample(self, batch, rng):
        ix = rng.integers(0, len(self), batch)
        return self.sa[ix], self.r[ix], self.nx[ix], self.nm[ix], self.done[ix]


class DQNAgent:
    def __init__(self, seed=0, lr=config.LEARNING_RATE, gamma=config.GAMMA):
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.q = QNetwork()
        self.target = QNetwork()
        self.target.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.gamma = gamma
        self.memory = ReplayMemory()
        self.steps = 0

    @torch.no_grad()
    def act(self, feats, mask, epsilon=0.0):
        ok = np.flatnonzero(mask)
        if self.rng.random() < epsilon:
            return int(self.rng.choice(ok))
        q = self.q(torch.from_numpy(feats)).numpy()
        q[~mask] = -np.inf
        return int(np.argmax(q))

    def learn(self, batch=config.BATCH_SIZE):
        sa, r, nx, nm, done = map(torch.from_numpy, self.memory.sample(batch, self.rng))
        q_sa = self.q(sa)
        with torch.no_grad():
            qn = self.target(nx)                       # [B, max_servers]
            qn = qn.masked_fill(~nm, -1e9).max(dim=1).values
            qn = torch.where(nm.any(dim=1), qn, torch.zeros_like(qn))
            y = r + self.gamma * (1.0 - done) * qn
        loss = nn.functional.smooth_l1_loss(q_sa, y)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.opt.step()
        return loss.item()

    def sync_target(self):
        self.target.load_state_dict(self.q.state_dict())

    def save(self, path, meta=None):
        torch.save({"state_dict": self.q.state_dict(), "meta": meta or {}}, path)

    @staticmethod
    def load(path):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        agent = DQNAgent()
        agent.q.load_state_dict(ckpt["state_dict"])
        agent.q.eval()
        agent.meta = ckpt.get("meta", {})
        return agent


class DQNScheduler:
    """Wraps a trained agent so it can be used like the baseline schedulers."""

    def __init__(self, agent, name):
        self.agent, self.name = agent, name

    def select(self, env, idx):
        feats, mask, _ = env.candidates(idx)
        return self.agent.act(feats, mask, epsilon=0.0)
