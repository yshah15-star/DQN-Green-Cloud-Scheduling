"""
Final experiments: every scheduler on identical, unseen TEST workloads.

Scenarios  : light, medium, heavy (trace windows) and burst (synthetic arrivals, trace task sizes)
             on the 10-server mixed cluster, TEST_WINDOWS windows each.
Generalisation: the medium scenario on 5 and 20 servers (models were never trained for a
             fixed size - this checks the size-independent design).
DQN models : every trained seed is evaluated on every window (weights frozen, epsilon = 0).
Random     : sanity-check reference only.

Outputs: results/raw_results.csv, results/summary.csv, data/processed/test_workloads/*.csv

Usage: python -m src.run_experiments            (full)
       python -m src.run_experiments --quick    (1 window, 2 scenarios, seed-0 models: ~1 minute)
"""
import argparse
import time

import pandas as pd

import config
from src.cloud_environment import Workload, build_cluster
from src.dqn_agent import DQNAgent, DQNScheduler
from src.schedulers import FCFS, MinMin, RandomScheduler, RoundRobin, run_episode
from src.workloads import make_workload

ORDER = ["FCFS", "Round Robin", "Min-Min", "Energy-focused DQN", "Multi-objective DQN", "Random"]
MODEL_FILES = {"Energy-focused DQN": "energy_dqn", "Multi-objective DQN": "multi_dqn_balanced"}


def load_models(seeds):
    models = {}
    for label, stem in MODEL_FILES.items():
        models[label] = []
        for s in seeds:
            p = config.MODELS_DIR / f"{stem}_seed{s}.pth"
            if p.exists():
                models[label].append((s, DQNAgent.load(p)))
        if not models[label]:
            print(f"warning: no trained model found for {label} ({stem}_seed*.pth)")
    return models


def evaluate(scenario, n_servers, window, models, save_workload=True):
    types = build_cluster(n_servers, "mixed")
    df = make_workload(scenario, config.TEST_TASKS, seed=10_000 + window, split="test")
    if save_workload and n_servers == config.TEST_SERVERS:
        out = config.DATA_PROCESSED / "test_workloads"
        out.mkdir(parents=True, exist_ok=True)
        df.to_csv(out / f"{scenario}_window{window}.csv", index=False)
    w = Workload(df)
    rows = []
    runs = [(s.name, None, s) for s in (FCFS(), RoundRobin(), MinMin(), RandomScheduler(window))]
    for label, agents in models.items():
        runs += [(label, seed, DQNScheduler(agent, label)) for seed, agent in agents]
    for name, seed, sched in runs:
        m, _ = run_episode(types, w, sched)
        rows.append(dict(scenario=scenario, servers=n_servers, window=window, scheduler=name,
                         model_seed=seed, **m))
    return rows


def main(quick=False):
    t0 = time.time()
    seeds = config.TRAIN_SEEDS[:1] if quick else config.TRAIN_SEEDS
    models = load_models(seeds)
    windows = range(1 if quick else config.TEST_WINDOWS)
    scenarios = ["light", "medium"] if quick else ["light", "medium", "heavy", "burst"]
    rows = []
    for sc in scenarios:
        for k in windows:
            rows += evaluate(sc, config.TEST_SERVERS, k, models)
        print(f"{sc}: done ({time.time() - t0:.0f}s)", flush=True)
    if not quick:
        for n in config.GENERALIZATION_SERVERS:
            for k in windows:
                rows += evaluate("medium", n, k, models, save_workload=False)
            print(f"generalisation {n} servers: done ({time.time() - t0:.0f}s)", flush=True)

    raw = pd.DataFrame(rows)
    suffix = "_quick" if quick else ""
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    raw.to_csv(config.RESULTS_DIR / f"raw_results{suffix}.csv", index=False)
    metrics = ["energy_kwh", "makespan_s", "active_utilization", "load_imbalance",
               "sla_violation_pct", "avg_wait_s", "avg_response_s"]
    g = raw.groupby(["scenario", "servers", "scheduler"])[metrics]
    summary = g.mean().add_suffix("_mean").join(g.std(ddof=0).add_suffix("_sd")).join(g.size().rename("runs"))
    summary = summary.reset_index()
    summary["order"] = summary["scheduler"].map({n: i for i, n in enumerate(ORDER)})
    summary = summary.sort_values(["servers", "scenario", "order"]).drop(columns="order")
    summary.to_csv(config.RESULTS_DIR / f"summary{suffix}.csv", index=False)
    pd.set_option("display.width", 200)
    print(summary[["scenario", "servers", "scheduler", "energy_kwh_mean", "makespan_s_mean",
                   "active_utilization_mean", "load_imbalance_mean", "sla_violation_pct_mean", "runs"]]
          .round(3).to_string(index=False))
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(ap.parse_args().quick)
