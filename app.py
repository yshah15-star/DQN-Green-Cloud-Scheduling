"""
Streamlit demo: change the inputs, press Run, and compare all five schedulers on the same workload.

Run:  streamlit run app.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import config
from src.cloud_environment import Workload, build_cluster
from src.dqn_agent import DQNAgent, DQNScheduler
from src.schedulers import FCFS, MinMin, RoundRobin, run_episode
from src.workloads import make_workload

COLORS = {"FCFS": "#9aa5b1", "Round Robin": "#c3a37a", "Min-Min": "#7f8fa6",
          "Energy-focused DQN": "#2e8b57", "Multi-objective DQN": "#1f5fa8"}
LOWER_IS_BETTER = {"Energy (kWh)": True, "Makespan (s)": True, "Active utilization": False,
                   "Load imbalance": True, "SLA violations (%)": True, "Avg wait (s)": True}

st.set_page_config(page_title="DQN Green Cloud Scheduling", layout="wide")
st.title("Energy-efficient task scheduling: FCFS vs Round Robin vs Min-Min vs DQN")
st.caption("COMP 5112 Research Methodology · Lakehead University · Workloads from Alibaba Cluster Trace v2018 (test split)")


@st.cache_resource
def load_agent(stem, seed):
    p = config.MODELS_DIR / f"{stem}_seed{seed}.pth"
    return DQNAgent.load(p) if p.exists() else None


with st.sidebar:
    st.header("Inputs")
    n_tasks = st.slider("Number of tasks", 50, 1000, 300, step=50)
    n_servers = st.slider("Number of servers", 3, 30, 10)
    source = st.selectbox("Workload", ["light", "medium", "heavy", "burst"], index=1,
                          format_func=lambda s: {"light": "Trace: light", "medium": "Trace: medium",
                                                 "heavy": "Trace: heavy", "burst": "Synthetic burst"}[s])
    mix = st.selectbox("Server mix", ["mixed", "std", "perf", "eco"],
                       format_func=lambda m: {"mixed": "Mixed (std / eco / perf)", "std": "All standard",
                                              "perf": "All performance", "eco": "All eco"}[m])
    seed = st.number_input("Random seed (selects the trace window)", 0, 10_000, 0)
    model_seed = st.selectbox("DQN training seed", config.TRAIN_SEEDS)
    presets = [p for p in config.REWARD_PRESETS if (config.MODELS_DIR / f"multi_dqn_{p}_seed{model_seed}.pth").exists()]
    preset = st.selectbox("Multi-objective preset", presets or ["balanced"])
    run = st.button("Run simulation", type="primary", width="stretch")
    st.caption("Arrival rates are calibrated to a 10-server mixed cluster, so fewer servers means "
               "higher real load and more servers means lower load.")

if run:
    types = build_cluster(n_servers, mix)
    df = make_workload(source, n_tasks, seed=int(seed), split="test")
    w = Workload(df)
    scheds = [FCFS(), RoundRobin(), MinMin()]
    e_agent = load_agent("energy_dqn", model_seed)
    m_agent = load_agent(f"multi_dqn_{preset}", model_seed)
    if e_agent:
        scheds.append(DQNScheduler(e_agent, "Energy-focused DQN"))
    if m_agent:
        scheds.append(DQNScheduler(m_agent, "Multi-objective DQN"))
    if not (e_agent and m_agent):
        st.warning("Trained DQN model files not found in models/. Run training first (see README).")

    rows, envs = [], {}
    with st.spinner("Simulating..."):
        for s in scheds:
            m, env = run_episode(types, w, s, record_timeline=True)
            envs[s.name] = env
            rows.append({"Scheduler": s.name, "Energy (kWh)": m["energy_kwh"], "Makespan (s)": m["makespan_s"],
                         "Active utilization": m["active_utilization"], "Load imbalance": m["load_imbalance"],
                         "SLA violations (%)": m["sla_violation_pct"], "Avg wait (s)": m["avg_wait_s"]})
    res = pd.DataFrame(rows).set_index("Scheduler")

    st.subheader("Results")
    c1, c2, c3 = st.columns(3)
    c1.metric("Tasks", n_tasks)
    c2.metric("Servers", f"{n_servers} ({mix})")
    c3.metric("Total task work (core-hours)", f"{(df.cpu_request * df.duration).sum() / 3600:.1f}")

    def highlight(col):
        best = col.min() if LOWER_IS_BETTER[col.name] else col.max()
        return ["font-weight: bold; background-color: rgba(46,139,87,0.18)" if v == best else "" for v in col]

    st.dataframe(res.style.apply(highlight).format({"Energy (kWh)": "{:.3f}", "Makespan (s)": "{:.0f}",
                                                    "Active utilization": "{:.2f}", "Load imbalance": "{:.3f}",
                                                    "SLA violations (%)": "{:.1f}", "Avg wait (s)": "{:.1f}"}),
                 width="stretch")
    st.caption("Best value per column is highlighted.")

    cols = st.columns(3)
    for i, metric in enumerate(["Energy (kWh)", "Makespan (s)", "Active utilization",
                                "Load imbalance", "SLA violations (%)", "Avg wait (s)"]):
        fig, ax = plt.subplots(figsize=(4.2, 2.8))
        ax.bar(range(len(res)), res[metric], color=[COLORS[n] for n in res.index])
        ax.set_xticks(range(len(res)))
        ax.set_xticklabels([n.replace(" DQN", "\nDQN").replace("Round ", "Round\n") for n in res.index], fontsize=7)
        ax.set_title(metric, loc="left", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        cols[i % 3].pyplot(fig)
        plt.close(fig)

    st.subheader("Cumulative energy over time")
    fig, ax = plt.subplots(figsize=(10, 3.4))
    for name, env in envs.items():
        tl = np.array(env.timeline)
        ax.plot(tl[:, 0] - tl[0, 0], tl[:, 1] / 3.6e6, label=name, color=COLORS[name], lw=1.6)
    ax.set_xlabel("Simulation time (s)")
    ax.set_ylabel("Energy (kWh)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Average utilization per server")
    util = pd.DataFrame({n: e.per_server_utilization() for n, e in envs.items()},
                        index=[f"S{i} {t.name}" for i, t in enumerate(types)])
    fig, ax = plt.subplots(figsize=(10, 3.4))
    width = 0.8 / len(util.columns)
    for k, name in enumerate(util.columns):
        ax.bar(np.arange(len(util)) + (k - len(util.columns) / 2 + 0.5) * width, util[name], width,
               label=name, color=COLORS[name])
    ax.set_xticks(range(len(util)))
    ax.set_xticklabels(util.index, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Time-averaged CPU utilization")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, ncol=5)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    with st.expander("Workload used in this run"):
        st.dataframe(df.head(200), width="stretch")
        st.caption("task_id, arrival_time (s), duration (s, at reference speed), cpu_request (cores), "
                   "memory_request (normalised), deadline (derived: arrival + duration × slack).")
else:
    st.info("Choose the inputs on the left and press **Run simulation**.")
