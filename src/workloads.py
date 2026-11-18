"""
Build episode workloads from the processed trace splits.

* A window of n consecutive tasks is taken from a split (train / val / test).
* Task sizes (duration, CPU, memory) and the relative arrival pattern come from the trace.
* Arrival times are rescaled (one common factor, so the trace's relative pattern is kept) until
  the LOAD FACTOR on the REFERENCE cluster equals the target level (config.LOAD_LEVELS):
      load factor = mean CPU demand of running tasks during the arrival period / cluster capacity
  where demand counts every task from its arrival for its duration. Changing the number of servers
  later therefore changes the real load, which is what the UI demonstrates.
* Burst workloads are synthetic arrivals (a base rate plus short bursts at a much higher rate) whose
  task sizes are sampled from the trace split, scaled to the same kind of load factor - stress test only.
* Deadlines are re-derived after rescaling: arrival + duration * SLACK_FACTOR.
"""
from functools import lru_cache

import numpy as np
import pandas as pd

import config
from src.cloud_environment import build_cluster, cluster_capacity


@lru_cache(maxsize=4)
def load_split(name):
    path = config.DATA_PROCESSED / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run: python -m src.preprocess_trace --input <batch_task file>")
    return pd.read_csv(path)


def _finish(df, arrivals):
    df = df.copy()
    df["arrival_time"] = arrivals - arrivals.min()
    df["deadline"] = df["arrival_time"] + df["duration"] * config.SLACK_FACTOR
    df["task_id"] = np.arange(len(df))
    return df.reset_index(drop=True)[["task_id", "arrival_time", "duration", "cpu_request",
                                      "memory_request", "deadline"]]


def reference_capacity():
    return cluster_capacity(build_cluster(config.REFERENCE_SERVERS, config.REFERENCE_MIX))


def load_factor(arr, dur, cpu, capacity):
    """Mean CPU demand during the arrival period [0, last arrival] divided by capacity."""
    T = max(arr.max(), 1e-9)
    busy = np.clip(np.minimum(arr + dur, T) - arr, 0, None)
    return float((cpu * busy).sum() / T / capacity)


def scale_to_load(arr, dur, cpu, load, capacity):
    """Find one scale factor for the arrival times that gives the target load factor (bisection)."""
    arr = arr / max(arr.max(), 1e-9)                 # normalised pattern on [0, 1]
    lo, hi = 1e-2, 1e7                               # span of the arrival period in seconds
    for _ in range(80):
        mid = np.sqrt(lo * hi)
        if load_factor(arr * mid, dur, cpu, capacity) > load:
            lo = mid                                 # too much demand -> spread arrivals out
        else:
            hi = mid
    return arr * np.sqrt(lo * hi)


def trace_window(split, n_tasks, load, rng, capacity=None):
    """n consecutive trace tasks, arrivals rescaled to the target load factor."""
    data = load_split(split)
    n_tasks = min(n_tasks, len(data))
    start = int(rng.integers(0, len(data) - n_tasks + 1))
    df = data.iloc[start:start + n_tasks]
    arr = df["arrival_time"].to_numpy(float)
    # spread tasks that share the same trace second, keeping the relative pattern
    arr = np.sort(arr - arr.min() + rng.uniform(0, 1, len(arr)))
    capacity = capacity or reference_capacity()
    arr = scale_to_load(arr, df["duration"].to_numpy(float), df["cpu_request"].to_numpy(float), load, capacity)
    return _finish(df, arr)


def burst_window(split, n_tasks, rng, capacity=None):
    """Synthetic burst arrivals; task sizes bootstrapped from the trace split."""
    data = load_split(split)
    df = data.sample(n=n_tasks, replace=True, random_state=int(rng.integers(1 << 31)))
    capacity = capacity or reference_capacity()
    p = config.BURST
    n_burst = int(n_tasks * p["burst_fraction"])
    n_base = n_tasks - n_burst
    base = np.cumsum(rng.exponential(1.0, n_base))                    # base rate 1 (relative units)
    centers = rng.uniform(0.1, 0.9, p["n_bursts"]) * base[-1]
    per = np.array_split(np.arange(n_burst), p["n_bursts"])
    bursts = [c + np.cumsum(rng.exponential(1.0 / p["burst_rate_ratio"], len(ix))) for c, ix in zip(centers, per)]
    arr = np.concatenate([base] + bursts)
    order = np.argsort(arr)
    df = df.iloc[order]
    arr = scale_to_load(arr[order] - arr.min(), df["duration"].to_numpy(float),
                        df["cpu_request"].to_numpy(float), p["mean_load"], capacity)
    return _finish(df, arr)


def make_workload(source, n_tasks, seed, split="test", capacity=None):
    rng = np.random.default_rng(seed)
    if source == "burst":
        return burst_window(split, n_tasks, rng, capacity)
    return trace_window(split, n_tasks, config.LOAD_LEVELS[source], rng, capacity)
