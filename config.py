"""
Central configuration for the DQN Green Cloud Scheduling project.
Every experiment parameter lives here so results can be reproduced exactly.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# GREEN_WORKDIR lets a scratch copy of data/models/results live elsewhere (used only for dry runs)
WORK = Path(os.environ.get("GREEN_WORKDIR", ROOT))
DATA_RAW = WORK / "data" / "raw"
DATA_PROCESSED = WORK / "data" / "processed"
MODELS_DIR = WORK / "models"
RESULTS_DIR = WORK / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

# ---------------------------------------------------------------------------
# Dataset: Alibaba Cluster Trace v2018, batch_task table
# ---------------------------------------------------------------------------
TRACE_URL = "http://aliopentrace.oss-cn-beijing.aliyuncs.com/v2018Traces/batch_task.tar.gz"
TRACE_REPO = "https://github.com/alibaba/clusterdata/tree/master/cluster-trace-v2018"
BATCH_TASK_COLUMNS = ["task_name", "instance_num", "job_name", "task_type", "status",
                      "start_time", "end_time", "plan_cpu", "plan_mem"]
SAMPLE_ROWS = 200_000          # tasks kept after filtering (uniform random sample)
MIN_DURATION_S = 10            # drop very short tasks
MAX_DURATION_S = 3600          # drop tasks longer than 1 hour (avoids a few tasks dominating makespan)
MAX_TASK_CORES = 16            # drop tasks needing more cores than the smallest server holds
SPLIT = (0.70, 0.15, 0.15)     # time-based train / validation / test split
SLACK_FACTOR = 2.0             # deadline = arrival + duration * SLACK_FACTOR (derived, not in trace)

# ---------------------------------------------------------------------------
# Server types (heterogeneous). Power in watts. Memory in Alibaba normalised units, where a
# 96-core Alibaba machine = 100, so a server's memory = cores * 100 / 96.
# Sizes fit the real trace (median task 0.5 cores, max 10 cores / 8.2 memory units).
# perf = fast but power-hungry, std = middle, eco = slow but frugal.
# ---------------------------------------------------------------------------
SERVER_TYPES = {
    "perf": dict(cores=24, mem=25.0, speed=1.3, p_idle=200.0, p_max=400.0, p_sleep=15.0),
    "std":  dict(cores=16, mem=16.7, speed=1.0, p_idle=120.0, p_max=260.0, p_sleep=10.0),
    "eco":  dict(cores=16, mem=16.7, speed=0.7, p_idle=60.0,  p_max=150.0, p_sleep=5.0),
}
MIXED_PATTERN = ["std", "eco", "perf", "std", "eco"]   # repeated to build an n-server mixed cluster
REFERENCE_SERVERS = 10          # workload arrival rate is calibrated to this cluster size
REFERENCE_MIX = "mixed"

# Workload scenarios. Load factor = mean CPU demand of running tasks during the arrival period
# divided by the reference cluster's capacity (see workloads.py).
LOAD_LEVELS = {"light": 0.3, "medium": 0.6, "heavy": 0.9}
# Burst: base arrivals plus 3 bursts arriving 8x faster; whole pattern scaled to a mean load of 0.6
BURST = dict(mean_load=0.6, burst_rate_ratio=8.0, burst_fraction=0.3, n_bursts=3)

# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------
MINMIN_BATCH = 10               # batch-mode Min-Min: tasks are mapped in batches of this size

# ---------------------------------------------------------------------------
# DQN
# ---------------------------------------------------------------------------
MAX_SERVERS = 32                # padding size for replay memory (network itself is size-independent)
HIDDEN = (64, 64)
LEARNING_RATE = 1e-3            # defaults; per-model values selected on validation are in HPARAMS
GAMMA = 0.95
# Selected by the validation search (results/hparam_search_*.csv): lr in {1e-3, 3e-4} x gamma in {0.9, 0.95}
HPARAMS = {"energy": dict(lr=1e-3, gamma=0.90), "multi": dict(lr=1e-3, gamma=0.95)}
BATCH_SIZE = 64
REPLAY_SIZE = 40_000
WARMUP_STEPS = 1_000
TARGET_SYNC = 1_000             # steps between target-network updates
TRAIN_EVERY = 2                 # gradient step every N environment steps
EPS_START, EPS_END = 1.0, 0.05
EPS_DECAY_FRACTION = 0.6        # epsilon reaches EPS_END after this fraction of training steps
EPISODES = 400
TASKS_PER_EPISODE = 150
TRAIN_SERVERS_RANGE = (4, 14)   # random cluster size per training episode
TRAIN_LOAD_RANGE = (0.2, 1.1)
VALIDATE_EVERY = 10             # episodes
VAL_EPISODES = 12
MIN_SELECT_EPISODE = 100       # checkpoints are only kept after most of the exploration phase
VAL_TASKS = 200
REWARD_CLIP = 10.0

# Multi-objective reward weights (Balanced preset) for normalized dE, dMakespan, dLoadImbalance
REWARD_PRESETS = {
    "balanced":     (1 / 3, 1 / 3, 1 / 3),
    "energy_first": (0.6, 0.2, 0.2),    # optional
    "speed_first":  (0.2, 0.6, 0.2),    # optional
}
P_REF = 300.0                   # W, used to normalise energy deltas (dimensionless reward)

TRAIN_SEEDS = [0, 1, 2, 3, 4]   # each DQN is trained once per seed

# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------
TEST_TASKS = 300
TEST_SERVERS = 10
TEST_WINDOWS = 5                # distinct test windows (seeds) per scenario
GENERALIZATION_SERVERS = [5, 20]
