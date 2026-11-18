"""
Discrete-event cloud data-centre simulator.

Model
-----
* Tasks arrive online (arrival_time, duration at reference speed, cpu cores, memory).
* A scheduler assigns each arriving task to one server.
* Each server runs tasks while it has free CPU cores and memory; otherwise the task waits
  in that server's FIFO queue.
* Execution time on a server = duration / server.speed.
* A server with no running or queued task sleeps (low power); otherwise it is awake and draws
  P_idle + (P_max - P_idle) * utilization (see energy_model.py).

The same environment is used by every scheduler, so all of them see identical workloads.
"""
import heapq
from collections import deque
from dataclasses import dataclass

import numpy as np

import config
from src import energy_model as em

EPS = 1e-9
N_FEATURES = 21


@dataclass(frozen=True)
class ServerType:
    name: str
    cores: int
    mem: float
    speed: float
    p_idle: float
    p_max: float
    p_sleep: float


def server_type(name):
    return ServerType(name=name, **config.SERVER_TYPES[name])


def build_cluster(n_servers, mix="mixed"):
    """Return a list of ServerType for an n-server cluster. mix: 'mixed' or a type name."""
    if mix == "mixed":
        pattern = config.MIXED_PATTERN
        return [server_type(pattern[i % len(pattern)]) for i in range(n_servers)]
    return [server_type(mix) for _ in range(n_servers)]


def cluster_capacity(types):
    """Effective capacity in reference core-seconds per second."""
    return sum(t.cores * t.speed for t in types)


class Server:
    def __init__(self, sid, stype):
        self.id = sid
        self.t = stype
        self.used_cpu = 0.0
        self.used_mem = 0.0
        self.running = []          # heap of (end_time, task_idx, cpu, mem)
        self.queue = deque()       # (task_idx, cpu, mem, exec_time)

    @property
    def awake(self):
        return bool(self.running) or bool(self.queue)

    @property
    def utilization(self):
        return self.used_cpu / self.t.cores

    def fits(self, cpu, mem):
        return (self.used_cpu + cpu <= self.t.cores + EPS) and (self.used_mem + mem <= self.t.mem + EPS)

    def can_host(self, cpu, mem):
        return cpu <= self.t.cores + EPS and mem <= self.t.mem + EPS


class Workload:
    """Arrays describing one episode's tasks (sorted by arrival)."""

    def __init__(self, df):
        df = df.sort_values("arrival_time", kind="stable").reset_index(drop=True)
        self.df = df
        self.arrival = df["arrival_time"].to_numpy(float)
        self.duration = df["duration"].to_numpy(float)
        self.cpu = df["cpu_request"].to_numpy(float)
        self.mem = df["memory_request"].to_numpy(float)
        self.deadline = df["deadline"].to_numpy(float)
        self.n = len(df)


class CloudEnv:
    def __init__(self, types, workload, record_timeline=False):
        self.types = types
        self.servers = [Server(i, t) for i, t in enumerate(types)]
        self.w = workload
        self.time = float(workload.arrival[0]) if workload.n else 0.0
        self.t0 = self.time
        self.energy_j = 0.0
        self.server_energy = np.zeros(len(types))
        self.start = np.full(workload.n, np.nan)
        self.finish_t = np.full(workload.n, np.nan)
        self.assigned = np.full(workload.n, -1, dtype=int)
        # time integrals for metrics
        self._int_used = 0.0         # sum of used cores over time
        self._int_awake_cores = 0.0  # cores of awake servers over time
        self._int_std = 0.0          # std of per-server utilization over time
        self._int_util = np.zeros(len(types))
        self.record_timeline = record_timeline
        self.timeline = [(self.time, 0.0)] if record_timeline else None

    # ------------------------------------------------------------------ time
    def _integrate(self, t_new):
        dt = t_new - self.time
        if dt <= 0:
            return
        utils = np.empty(len(self.servers))
        used = awake_cores = 0.0
        for j, s in enumerate(self.servers):
            u = s.utilization
            utils[j] = u
            e = em.energy_joules(em.server_power(s.t, u, s.awake), dt)
            self.server_energy[j] += e
            self.energy_j += e
            used += s.used_cpu
            if s.awake:
                awake_cores += s.t.cores
        self._int_used += used * dt
        self._int_awake_cores += awake_cores * dt
        self._int_std += utils.std() * dt
        self._int_util += utils * dt
        self.time = t_new
        if self.record_timeline:
            self.timeline.append((t_new, self.energy_j))

    def _start(self, s, idx, cpu, mem, exec_time):
        s.used_cpu += cpu
        s.used_mem += mem
        end = self.time + exec_time
        heapq.heappush(s.running, (end, idx, cpu, mem))
        self.start[idx] = self.time
        self.finish_t[idx] = end

    def _drain_queue(self, s):
        while s.queue and s.fits(s.queue[0][1], s.queue[0][2]):
            idx, cpu, mem, ex = s.queue.popleft()
            self._start(s, idx, cpu, mem, ex)

    def advance(self, t_target):
        """Run the simulation forward to time t_target, processing completions."""
        while True:
            nxt, srv = np.inf, None
            for s in self.servers:
                if s.running and s.running[0][0] < nxt:
                    nxt, srv = s.running[0][0], s
            if srv is None or nxt > t_target:
                break
            self._integrate(nxt)
            while srv.running and srv.running[0][0] <= nxt + EPS:
                _, _, cpu, mem = heapq.heappop(srv.running)
                srv.used_cpu = max(0.0, srv.used_cpu - cpu)
                srv.used_mem = max(0.0, srv.used_mem - mem)
            self._drain_queue(srv)
        if np.isfinite(t_target):
            self._integrate(t_target)

    def assign(self, idx, sid):
        s = self.servers[sid]
        cpu, mem = self.w.cpu[idx], self.w.mem[idx]
        ex = self.w.duration[idx] / s.t.speed
        self.assigned[idx] = sid
        if not s.queue and s.fits(cpu, mem):
            self._start(s, idx, cpu, mem, ex)
        else:
            s.queue.append((idx, cpu, mem, ex))

    def finish(self):
        self.advance(np.inf)

    # ------------------------------------------------------------ projection
    def project(self, s, extra=None):
        """
        Predict a server's future assuming no further arrivals.
        extra = (cpu, mem, exec_time) of a candidate task appended to the queue.
        Returns (start_of_extra, finish_of_extra, empty_time, backlog_seconds).
        """
        t = self.time
        ev = list(s.running)
        free_cpu, free_mem = s.t.cores - s.used_cpu, s.t.mem - s.used_mem
        q = list(s.queue)
        if extra is not None:
            q.append((-1, extra[0], extra[1], extra[2]))
        backlog = sum((e - t) * c for e, _, c, _ in ev) + sum(c * x for _, c, _, x in q)
        now, last_end, qi, start_extra = t, t, 0, None
        while True:
            while qi < len(q) and q[qi][1] <= free_cpu + EPS and q[qi][2] <= free_mem + EPS:
                _, c, m, x = q[qi]
                free_cpu -= c
                free_mem -= m
                heapq.heappush(ev, (now + x, -2, c, m))
                if extra is not None and qi == len(q) - 1:
                    start_extra = now
                qi += 1
            if not ev:
                break
            end, _, c, m = heapq.heappop(ev)
            now = max(now, end)
            last_end = max(last_end, end)
            free_cpu += c
            free_mem += m
        finish_extra = None if start_extra is None else start_extra + extra[2]
        return start_extra, finish_extra, last_end, backlog / s.t.cores

    def candidates(self, idx):
        """
        Features for every (task, server) pair plus the projected deltas used by the rewards.
        Returns feats [n, N_FEATURES], mask [n] (True = server can host the task), info dict.
        """
        n = len(self.servers)
        cpu, mem, dur = self.w.cpu[idx], self.w.mem[idx], self.w.duration[idx]
        t = self.time
        old = [self.project(s) for s in self.servers]
        old_empty = np.array([o[2] for o in old])
        old_backlog = np.array([o[3] for o in old])
        old_M = old_empty.max() if n else t
        old_L = old_backlog.std()
        utils = np.array([s.utilization for s in self.servers])
        awake = np.array([s.awake for s in self.servers], dtype=float)

        feats = np.zeros((n, N_FEATURES), dtype=np.float32)
        mask = np.zeros(n, dtype=bool)
        dE = np.full(n, np.nan)
        dT = np.full(n, np.nan)
        dL = np.full(n, np.nan)
        dnorm = max(dur, 1.0)
        for j, s in enumerate(self.servers):
            if not s.can_host(cpu, mem):
                continue
            mask[j] = True
            ex = dur / s.t.speed
            st, fin, empty, back = self.project(s, (cpu, mem, ex))
            ext = empty - max(t, old_empty[j])
            dE[j] = em.marginal_energy(s.t, cpu, ex, ext)
            dT[j] = max(old_M, empty) - old_M
            b = old_backlog.copy()
            b[j] = back
            dL[j] = b.std() - old_L
            wait = st - t
            feats[j] = [
                cpu / 16.0, mem / 10.0, np.log1p(dur) / 10.0,                       # task
                s.t.speed, s.t.p_idle / 400.0, s.t.p_max / 400.0, s.t.cores / 64.0,  # server
                utils[j], s.used_mem / s.t.mem, awake[j], np.log1p(len(s.queue)) / 3.0,
                min(old_backlog[j] / dnorm, 10.0) / 10.0,
                min(wait / dnorm, 10.0) / 10.0, ex / dnorm,                           # action
                min(dE[j] / (config.P_REF * dnorm), 10.0),
                min(dT[j] / dnorm, 10.0),
                float(np.clip(dL[j] / dnorm, -10.0, 10.0)),
                1.0 if wait <= EPS else 0.0,
                utils.mean(), awake.mean(), n / 20.0,                                 # global
            ]
        info = dict(dE=dE, dT=dT, dL=dL, dnorm=dnorm)
        return feats, mask, info

    # --------------------------------------------------------------- metrics
    def metrics(self):
        w = self.w
        makespan = float(np.nanmax(self.finish_t) - w.arrival.min())
        return dict(
            energy_kwh=em.to_kwh(self.energy_j),
            makespan_s=makespan,
            active_utilization=self._int_used / max(self._int_awake_cores, EPS),
            load_imbalance=self._int_std / max(self.time - self.t0, EPS),
            sla_violation_pct=100.0 * float(np.mean(self.finish_t > w.deadline + EPS)),
            avg_wait_s=float(np.nanmean(self.start - w.arrival)),
            avg_response_s=float(np.nanmean(self.finish_t - w.arrival)),
        )

    def per_server_utilization(self):
        return self._int_util / max(self.time - self.t0, EPS)
