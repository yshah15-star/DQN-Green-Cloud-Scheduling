"""
Train the Energy-focused DQN or the Multi-objective DQN.

Per-step rewards (computed right after each task is assigned, from the simulator's projection):
  Energy-focused DQN : r = - dE / (P_REF * d)
  Multi-objective DQN: r = - ( w1 * dE / (P_REF * d) + w2 * dT / d + w3 * dL / d )
where dE = extra energy caused by the placement (J), dT = increase of the projected makespan (s),
dL = change of the load imbalance (std of per-server backlog, s), d = task duration (s).
Dividing by the task's own scale makes the terms dimensionless; for the multi-objective reward
each term is further divided by its mean absolute value under a random policy (calibrate_scales),
so the weights compare like with like:
  r = - ( w1 * e / s_E + w2 * t / s_T + w3 * l / s_L )

Episodes use the TRAIN split only. Every VALIDATE_EVERY episodes the greedy policy is scored on
fixed VALIDATION episodes (relative to FCFS on the same episodes); the best checkpoint from
episode MIN_SELECT_EPISODE onwards is kept.

Usage:
  python -m src.train --reward energy --seed 0
  python -m src.train --reward multi --preset balanced --seed 0
  python -m src.train --reward energy --search      (small learning-rate x gamma search on validation)
"""
import argparse
import csv
import itertools
import json
import time

import numpy as np

import config
from src.cloud_environment import CloudEnv, Workload, build_cluster, server_type
from src.dqn_agent import DQNAgent, DQNScheduler
from src.schedulers import FCFS, RandomScheduler, run_episode
from src.workloads import trace_window


def reward_terms(info, a):
    """Dimensionless per-step terms: energy, makespan increase, load-imbalance change."""
    d = info["dnorm"]
    return info["dE"][a] / (config.P_REF * d), info["dT"][a] / d, info["dL"][a] / d


def calibrate_scales(n_episodes=20, seed=999):
    """
    Normalisation for the multi-objective reward: each term is divided by its mean absolute
    value under a random policy on TRAINING episodes, so the weights w1, w2, w3 compare
    like with like (otherwise the energy term is ~10-20x larger than the others).
    """
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_episodes):
        types, w = random_episode(rng)
        env = CloudEnv(types, w)
        for idx in range(w.n):
            env.advance(w.arrival[idx])
            _, mask, info = env.candidates(idx)
            a = int(rng.choice(np.flatnonzero(mask)))
            vals.append(reward_terms(info, a))
            env.assign(idx, a)
    return tuple(float(max(np.mean(np.abs(v)), 1e-6)) for v in np.array(vals).T)


def step_reward(info, a, mode, weights, scales):
    e, t, l = reward_terms(info, a)
    if mode == "energy":
        r = -e
    else:
        (w1, w2, w3), (se, st, sl) = weights, scales
        r = -(w1 * e / se + w2 * t / st + w3 * l / sl)
    return float(np.clip(r, -config.REWARD_CLIP, config.REWARD_CLIP))


def random_episode(rng, split="train", n_tasks=config.TASKS_PER_EPISODE):
    n = int(rng.integers(config.TRAIN_SERVERS_RANGE[0], config.TRAIN_SERVERS_RANGE[1] + 1))
    mix = "mixed" if rng.random() < 0.8 else str(rng.choice(list(config.SERVER_TYPES)))
    load = float(rng.uniform(*config.TRAIN_LOAD_RANGE))
    return build_cluster(n, mix), Workload(trace_window(split, n_tasks, load, rng))


def validation_set(seed=1234):
    rng = np.random.default_rng(seed)
    loads = list(config.LOAD_LEVELS.values())
    eps = []
    for k in range(config.VAL_EPISODES):
        types = build_cluster(config.TEST_SERVERS, "mixed")
        w = Workload(trace_window("val", config.VAL_TASKS, loads[k % len(loads)], rng))
        eps.append((types, w, run_episode(types, w, FCFS())[0]))
    return eps


def score(metrics, ref, mode, weights):
    e = metrics["energy_kwh"] / ref["energy_kwh"]
    if mode == "energy":
        return e
    w1, w2, w3 = weights
    return (w1 * e + w2 * metrics["makespan_s"] / ref["makespan_s"]
            + w3 * metrics["load_imbalance"] / max(ref["load_imbalance"], 1e-6))


def validate(agent, val_eps, mode, weights):
    sched = DQNScheduler(agent, "dqn")
    return float(np.mean([score(run_episode(t, w, sched)[0], ref, mode, weights) for t, w, ref in val_eps]))


def train(mode, seed, preset="balanced", lr=config.LEARNING_RATE, gamma=config.GAMMA,
          episodes=config.EPISODES, save=True, tag=None, verbose=True):
    weights = config.REWARD_PRESETS[preset]
    rng = np.random.default_rng(seed)
    agent = DQNAgent(seed=seed, lr=lr, gamma=gamma)
    val_eps = validation_set()
    scales = calibrate_scales() if mode == "multi" else (1.0, 1.0, 1.0)
    rand_score = float(np.mean([score(run_episode(t, w, RandomScheduler(seed))[0], ref, mode, weights)
                                for t, w, ref in val_eps]))
    total_steps = episodes * config.TASKS_PER_EPISODE
    decay = config.EPS_DECAY_FRACTION * total_steps
    name = tag or ("energy_dqn" if mode == "energy" else f"multi_dqn_{preset}")
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log, best, t0 = [], np.inf, time.time()

    for ep in range(1, episodes + 1):
        types, w = random_episode(rng)
        env = CloudEnv(types, w)
        ep_reward, losses, prev = 0.0, [], None
        for idx in range(w.n):
            env.advance(w.arrival[idx])
            feats, mask, info = env.candidates(idx)
            if prev is not None:
                agent.memory.push(prev[0], prev[1], feats, mask, False)
            eps = max(config.EPS_END, config.EPS_START - (config.EPS_START - config.EPS_END) * agent.steps / decay)
            a = agent.act(feats, mask, eps)
            r = step_reward(info, a, mode, weights, scales)
            env.assign(idx, a)
            prev = (feats[a], r)
            ep_reward += r
            agent.steps += 1
            if len(agent.memory) >= config.WARMUP_STEPS and agent.steps % config.TRAIN_EVERY == 0:
                losses.append(agent.learn())
            if agent.steps % config.TARGET_SYNC == 0:
                agent.sync_target()
        agent.memory.push(prev[0], prev[1], None, None, True)

        row = dict(episode=ep, servers=len(types), reward=ep_reward / w.n,
                   loss=float(np.mean(losses)) if losses else np.nan, epsilon=eps, val_score=np.nan)
        if ep % config.VALIDATE_EVERY == 0 or ep == episodes:
            v = validate(agent, val_eps, mode, weights)
            row["val_score"] = v
            if v < best and ep >= min(config.MIN_SELECT_EPISODE, episodes):
                best = v
                if save:
                    agent.save(config.MODELS_DIR / f"{name}_seed{seed}.pth",
                               meta=dict(mode=mode, preset=preset, seed=seed, episode=ep, val_score=v,
                                         lr=lr, gamma=gamma, weights=weights, scales=scales))
            if verbose:
                print(f"[{name} s{seed}] ep {ep:4d}  reward {row['reward']:+.3f}  loss {row['loss']:.4f}  "
                      f"eps {eps:.2f}  val {v:.4f} (best {best:.4f}, random {rand_score:.4f})  "
                      f"{time.time() - t0:.0f}s", flush=True)
        log.append(row)

    if save:
        with open(config.RESULTS_DIR / f"training_log_{name}_seed{seed}.csv", "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(log[0]))
            wr.writeheader()
            wr.writerows(log)
        with open(config.RESULTS_DIR / f"training_summary_{name}_seed{seed}.json", "w") as f:
            json.dump(dict(best_val_score=best, random_val_score=rand_score, fcfs_val_score=1.0,
                           seconds=time.time() - t0, lr=lr, gamma=gamma, episodes=episodes,
                           reward_scales=scales), f, indent=2)
    return best, rand_score


def search(mode, preset, episodes):
    rows = []
    for lr, gamma in itertools.product([1e-3, 3e-4], [0.9, 0.95]):
        best, _ = train(mode, seed=100, preset=preset, lr=lr, gamma=gamma, episodes=episodes,
                        save=False, verbose=False)
        rows.append(dict(mode=mode, preset=preset, lr=lr, gamma=gamma, episodes=episodes, best_val_score=best))
        print(rows[-1], flush=True)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.RESULTS_DIR / f"hparam_search_{mode}.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reward", choices=["energy", "multi"], required=True)
    ap.add_argument("--preset", default="balanced", choices=list(config.REWARD_PRESETS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=config.EPISODES)
    ap.add_argument("--lr", type=float, default=None, help="default: config.HPARAMS[reward]")
    ap.add_argument("--gamma", type=float, default=None, help="default: config.HPARAMS[reward]")
    ap.add_argument("--search", action="store_true")
    args = ap.parse_args()
    if args.search:
        search(args.reward, args.preset, args.episodes)
    else:
        hp = config.HPARAMS[args.reward]
        train(args.reward, args.seed, args.preset, args.lr or hp["lr"], args.gamma or hp["gamma"], args.episodes)
