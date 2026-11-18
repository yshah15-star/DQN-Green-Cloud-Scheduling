# DQN-Based Energy-Efficient Task Scheduling for Green Cloud Data Centres

COMP 5112 Research Methodology in Computer Science · Lakehead University · Fall 2026 · Professor: Dr. Jinan Fiaidhi
Team: Saravanan Thiruvengadam (1340909), Yash Ajaykumar Shah (1372718)

**Research question.** Can a DQN-based task scheduler improve the balance between energy consumption and system
performance compared with conventional methods in a simulated cloud data centre?

We compare five schedulers on identical, unseen workloads taken from the **Alibaba Cluster Trace v2018**:
FCFS, Round Robin, Min-Min, an **Energy-focused DQN** and a **Multi-objective DQN** (our improvement).

**Demo video:** _link to be added_

---

## 1. Quick start

```bash
git clone https://github.com/yshah15-star/DQN-Green-Cloud-Scheduling.git
cd DQN-Green-Cloud-Scheduling
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# a) Evaluate the included trained models on test workloads (under a minute)
python -m src.run_experiments --quick

# b) Launch the interactive UI
streamlit run app.py
```

Everything needed is in the repository: the processed dataset (`data/processed/`), the trained models (`models/`)
and the final results (`results/`). Nothing has to be retrained.

In the UI, choose the number of tasks, number of servers, workload (trace light / medium / heavy or synthetic burst),
server mix and random seed, then press **Run simulation**. All five schedulers run on the same workload and the page
shows a metrics table, comparison charts, cumulative energy over time and per-server utilization.

## 2. Full reproduction (about 30–40 minutes on a laptop CPU)

```bash
bash run_all.sh data/raw/batch_task_sample.csv
```

or step by step:

| Step | Command | Output |
| --- | --- | --- |
| Preprocess trace | `python -m src.preprocess_trace --input data/raw/batch_task_sample.csv` | `data/processed/{train,val,test}.csv`, `preprocess_report.json` |
| Hyper-parameter search | `python -m src.train --reward energy --search --episodes 200` (and `--reward multi`) | `results/hparam_search_*.csv` |
| Train Energy-focused DQN | `python -m src.train --reward energy --seed 0` (seeds 0–4) | `models/energy_dqn_seed*.pth`, `results/training_log_*.csv` |
| Train Multi-objective DQN | `python -m src.train --reward multi --preset balanced --seed 0` (seeds 0–4) | `models/multi_dqn_balanced_seed*.pth` |
| Experiments | `python -m src.run_experiments` | `results/raw_results.csv`, `results/summary.csv`, `data/processed/test_workloads/` |
| Figures | `python -m src.plots` | `results/figures/*.png` |

To start from the full trace instead of the included sample, download `batch_task.tar.gz` (link below) and pass
it to `--input`. All parameters are in `config.py`; random seeds are fixed.

## 3. Repository layout

```
├── README.md  requirements.txt  config.py  run_all.sh
├── app.py                         Streamlit UI
├── data/raw/                      Alibaba batch_task sample + SOURCE.md
├── data/processed/                train / val / test splits, test workloads used in the experiments
├── src/
│   ├── preprocess_trace.py        trace -> clean CSVs, time-based split, derived deadlines
│   ├── workloads.py               episode workloads (trace windows, load levels, synthetic bursts)
│   ├── cloud_environment.py       discrete-event simulator (Task, Server, CloudEnv), DQN features
│   ├── energy_model.py            power and energy model (the only place energy is calculated)
│   ├── schedulers.py              FCFS, Round Robin, Min-Min, Random + episode runner
│   ├── dqn_agent.py               size-independent DQN, replay memory, target network
│   ├── train.py                   training, rewards, validation, hyper-parameter search
│   ├── run_experiments.py         final evaluation on unseen test workloads
│   └── plots.py                   figures
├── models/                        trained DQN weights (5 seeds per model)
├── results/                       CSV results, training logs, figures/
└── references/sources.md          papers, dataset and software sources
```

## 4. Dataset

**Alibaba Cluster Trace v2018**, `batch_task` table — production batch tasks from a cluster of about 4,000 machines
over 8 days. Repository: https://github.com/alibaba/clusterdata/tree/master/cluster-trace-v2018 ·
file: http://aliopentrace.oss-cn-beijing.aliyuncs.com/v2018Traces/batch_task.tar.gz (125 MB).
The 20 GB `batch_instance` table is not used.

Preprocessing (`src/preprocess_trace.py`):
1. keep `status == Terminated` tasks with valid times and positive `plan_cpu`, `plan_mem`;
2. `duration = end_time − start_time`, kept between 10 s and 1 h;
3. `cpu_request = plan_cpu / 100` cores (kept ≤ 16 cores); `memory_request = plan_mem` (normalised, machine = 100);
4. uniform random sample of 200,000 tasks, sorted by time; `arrival_time = start_time` shifted to 0;
5. **time-based split 70 / 15 / 15** — earliest tasks train the DQNs, the next part is for validation (model and
   hyper-parameter selection), the latest part is the held-out test set. A time split avoids leakage between sets;
6. **derived deadline** = arrival + duration × 2.0 (slack factor). Deadlines are *not* in the trace.

Workload scenarios (`src/workloads.py`): a window of consecutive trace tasks keeps the real task sizes and relative
arrival pattern. Arrival times are rescaled by one common factor until the **load factor** on the reference
10-server mixed cluster reaches the target — light 0.3, medium 0.6, heavy 0.9 — where load factor = mean CPU demand
of running tasks during the arrival period ÷ cluster capacity. The **burst** scenario uses synthetic arrivals (a base
rate plus three bursts arriving 8× faster, mean load 0.6) with task sizes sampled from the trace; it is a stress test only.

## 5. Method

**Simulator.** Online, discrete-event. Each arriving task is assigned to one server; it runs if the server has free
cores and memory, otherwise it waits in that server's FIFO queue. Run time = duration / server speed.

**Servers (heterogeneous).**

| Type | Cores | Memory | Speed | P_idle (W) | P_max (W) | P_sleep (W) |
| --- | --- | --- | --- | --- | --- | --- |
| perf | 24 | 25.0 | 1.3 | 200 | 400 | 15 |
| std | 16 | 16.7 | 1.0 | 120 | 260 | 10 |
| eco | 16 | 16.7 | 0.7 | 60 | 150 | 5 |

Memory is in the trace's normalised units (a 96-core Alibaba machine = 100). Sizes were chosen to fit the real tasks
(median 0.5 cores, largest 10 cores). The mixed cluster repeats std, eco, perf, std, eco.

**Energy model** (`src/energy_model.py`): P(u) = P_idle + (P_max − P_idle)·u while a server has work; P_sleep when it
has none. Energy = ∫P dt, reported in kWh. Packing tasks onto fewer servers lets others sleep (saves energy);
spreading them finishes work sooner — the trade-off the DQNs learn.

**Baselines.** FCFS — first server (fixed order) that can start the task now, else the one where it starts earliest.
Round Robin — servers in rotation. Min-Min (Braun et al., 2001) — batch mode, 10 tasks per batch; repeatedly assigns
the task–server pair with the smallest expected completion time.

**DQN (our design, PyTorch).** Scheduling is an MDP: state = current task + cluster status; action = the server;
reward after each assignment. The network is **size-independent**: one shared MLP (21 → 64 → 64 → 1) scores every
candidate server, Q(state, server_i), from task, server, projected-effect and global features; the highest feasible
score wins. The same model therefore runs on 5, 10 or 20 servers. Standard DQN parts: experience replay (40k),
ε-greedy (1.0 → 0.05), target network (sync every 1,000 steps), Huber loss, Adam. Learning rate and γ were chosen
on the validation set from {1e-3, 3e-4} × {0.9, 0.95} (`results/hparam_search_*.csv`): Energy-focused DQN lr 1e-3,
γ 0.9; Multi-objective DQN lr 1e-3, γ 0.95. 400 training episodes of 150 tasks; episodes vary cluster size (4–14
servers), server mix and load (0.2–1.1). The checkpoint with the best validation score is kept.

**Per-step rewards** (from the simulator's projection of the assignment's effect; d = task duration):

- Energy-focused DQN: r = −ΔE / (P_ref · d)
- Multi-objective DQN: r = −( w1·ΔE/(P_ref·d)/s_E + w2·ΔT/d/s_T + w3·ΔL/d/s_L ), balanced w = (1/3, 1/3, 1/3)

ΔE = extra energy caused by the placement (dynamic share + extra awake time), ΔT = increase of the projected
makespan, ΔL = change of load imbalance (std of per-server backlog). s_E, s_T, s_L are each term's mean absolute
value under a random policy on training data, so the three terms are on comparable scales (`calibrate_scales`).

**Protocol.** All schedulers get exactly the same test workloads (4 scenarios × 5 windows, 300 tasks, 10 mixed
servers), plus a generalisation check on 5 and 20 servers. Each DQN is trained with 5 seeds; every trained model is
evaluated on every window with frozen weights. Results are mean ± SD.

**Metrics.** Energy (kWh) · makespan (s) · active CPU utilization (used cores / cores of awake servers) · load
imbalance (time-averaged std of server utilization) · SLA violations (% of tasks finishing after their deadline).

## Results

Held-out **test** workloads, 10 mixed servers, 300 tasks, 5 windows per scenario; DQN rows average 5 trained seeds ×
5 windows (25 runs). Mean ± SD. Full numbers: `results/summary.csv`; every run: `results/raw_results.csv`.

**Medium load (load factor 0.6)**

| Scheduler | Energy (kWh) | Makespan (s) | Active utilization | Load imbalance | SLA violations (%) |
| --- | --- | --- | --- | --- | --- |
| FCFS | 0.411 ± 0.077 | 2763 ± 1116 | 0.22 | 0.082 | 0.3 ± 0.7 |
| Round Robin | 0.437 ± 0.099 | 2680 ± 950 | 0.20 | 0.080 | 7.0 ± 4.4 |
| Min-Min | 0.277 ± 0.046 | 1792 ± 593 | 0.35 | 0.123 | 0.0 ± 0.0 |
| Energy-focused DQN | 0.263 ± 0.052 | 4150 ± 1402 | 0.57 | 0.194 | 85.8 ± 3.7 |
| Multi-objective DQN | 0.277 ± 0.059 | 1912 ± 799 | 0.37 | 0.116 | 15.1 ± 7.0 |

**Energy and makespan relative to FCFS, all scenarios**

| Scenario | Min-Min | Energy-focused DQN | Multi-objective DQN |
| --- | --- | --- | --- |
| Light (0.3) | energy −23%, makespan −29% | energy −23%, makespan +56% | energy −11%, makespan −21% |
| Medium (0.6) | energy −33%, makespan −35% | energy −36%, makespan +50% | energy −33%, makespan −31% |
| Heavy (0.9) | energy −31%, makespan −36% | energy −38%, makespan +54% | energy −38%, makespan −32% |
| Burst | energy −31%, makespan −33% | energy −41%, makespan +47% | energy −34%, makespan −23% |
| Medium, 5 servers | energy −21%, makespan −8% | energy −26%, makespan +113% | energy −20%, makespan −15% |
| Medium, 20 servers | energy −20%, makespan −42% | energy −28%, makespan +33% | energy −33%, makespan −39% |

**Findings**

1. **Both DQNs learn.** Every seed beats FCFS and random scheduling on its validation objective
   (`results/figures/validation_score.png`).
2. **Energy-focused DQN** uses the least energy in most scenarios (−23% to −41% vs FCFS) by packing tasks onto few
   servers, but this queues work: makespan rises 33–113% and about 85% of tasks miss their derived deadline.
   Optimising energy alone is not acceptable.
3. **Multi-objective DQN** keeps most of the energy saving (−33% to −38% vs FCFS at medium, heavy and burst load)
   while also cutting makespan (−15% to −39% across all scenarios). It matches Min-Min on medium-load energy and beats it at heavy load
   (−38% vs −31%) and on the 20-server cluster (energy −33% vs −20%).
4. **Trade-off:** the Multi-objective DQN misses more deadlines than Min-Min and FCFS (15% vs 0% at medium load),
   because deadlines are not part of its reward. Min-Min remains a very strong baseline.
5. **Size independence works:** the same trained models run on 5, 10 and 20 servers without retraining; the
   Multi-objective DQN's advantage is largest on the 20-server cluster.

Figures: `results/figures/` — energy, makespan, utilization, load balance, SLA violations, generalisation, training
reward / loss and validation curves.

## 6. Assumptions and limitations

- Simulation, not a real data centre; the power figures are representative values, not measurements.
- Arrival = trace `start_time` (the trace has no submission time); deadlines are derived (slack 2.0).
- Tasks are independent (the trace's DAG dependencies are ignored); each task is one schedulable unit using its
  per-instance `plan_cpu` / `plan_mem`.
- Servers wake instantly and with no extra energy; no network, storage or cooling energy.
- Arrival rates are rescaled to the chosen load levels; durations and resource sizes are kept from the trace.
- The DQN's features include the simulator's projection of each placement (its energy, makespan and imbalance
  effect), so it learns from engineered features rather than raw state. Because the Energy-focused DQN's reward
  equals one of these features, it reaches its final policy within the first ~100 episodes.
- Deadlines (SLA) are measured but not part of either reward, which explains the Multi-objective DQN's higher
  SLA-violation rate; adding a deadline term is the natural next step.
- Each test scenario uses 5 windows of 300 tasks; standard deviations are large because windows differ in task mix.

## 7. Attribution

All code (simulator, schedulers, DQN, training, experiments, UI) was written for this project from the algorithm
descriptions in the cited papers and the libraries' documentation; no external code was copied or adapted.
Full references: [`references/sources.md`](references/sources.md).
