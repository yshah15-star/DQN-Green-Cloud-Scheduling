"""
Baseline schedulers and the common episode runner.

FCFS        - tasks are handled in arrival order; each goes to the first server (fixed order)
              that can start it now; if none can, to the server where it would start earliest.
Round Robin - servers are taken in rotation, skipping any that cannot host the task.
Min-Min     - batch-mode Min-Min (Braun et al., 2001): tasks are collected in batches of
              MINMIN_BATCH; repeatedly the (task, server) pair with the smallest expected
              completion time is assigned, until the batch is empty.
Random      - uniform random feasible server (sanity-check baseline only).
"""
import numpy as np

import config
from src.cloud_environment import CloudEnv


class FCFS:
    name = "FCFS"

    def select(self, env, idx):
        cpu, mem = env.w.cpu[idx], env.w.mem[idx]
        best, best_start = None, np.inf
        for s in env.servers:
            if not s.can_host(cpu, mem):
                continue
            if not s.queue and s.fits(cpu, mem):
                return s.id
            st, _, _, _ = env.project(s, (cpu, mem, env.w.duration[idx] / s.t.speed))
            if st < best_start:
                best, best_start = s.id, st
        return best


class RoundRobin:
    name = "Round Robin"

    def __init__(self):
        self.ptr = 0

    def select(self, env, idx):
        n = len(env.servers)
        for k in range(n):
            s = env.servers[(self.ptr + k) % n]
            if s.can_host(env.w.cpu[idx], env.w.mem[idx]):
                self.ptr = (s.id + 1) % n
                return s.id
        raise RuntimeError("no server can host task")


class MinMin:
    name = "Min-Min"
    batch = config.MINMIN_BATCH

    def assign_batch(self, env, idxs):
        pending = list(idxs)
        while pending:
            best = (np.inf, None, None)
            for i in pending:
                for s in env.servers:
                    if not s.can_host(env.w.cpu[i], env.w.mem[i]):
                        continue
                    _, fin, _, _ = env.project(s, (env.w.cpu[i], env.w.mem[i], env.w.duration[i] / s.t.speed))
                    if fin < best[0]:
                        best = (fin, i, s.id)
            _, i, sid = best
            env.assign(i, sid)
            pending.remove(i)


class RandomScheduler:
    name = "Random"

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def select(self, env, idx):
        ok = [s.id for s in env.servers if s.can_host(env.w.cpu[idx], env.w.mem[idx])]
        return int(self.rng.choice(ok))


def run_episode(types, workload, scheduler, record_timeline=False):
    """Run one workload through one scheduler; return (metrics, env)."""
    env = CloudEnv(types, workload, record_timeline=record_timeline)
    w = workload
    if hasattr(scheduler, "assign_batch"):
        for b in range(0, w.n, scheduler.batch):
            idxs = list(range(b, min(b + scheduler.batch, w.n)))
            env.advance(w.arrival[idxs[-1]])      # batch closes when its last task has arrived
            scheduler.assign_batch(env, idxs)
    else:
        for idx in range(w.n):
            env.advance(w.arrival[idx])
            env.assign(idx, scheduler.select(env, idx))
    env.finish()
    return env.metrics(), env
