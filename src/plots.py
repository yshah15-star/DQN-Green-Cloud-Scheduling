"""
Figures for the PPT and paper (PNG, results/figures/).

  energy_comparison.png, makespan_comparison.png, utilization_comparison.png,
  load_balance_comparison.png, sla_violations_comparison.png   - mean +/- SD per scenario
  generalization_energy.png                                     - 5 / 10 / 20 servers, medium load
  training_reward.png, training_loss.png, validation_score.png  - learning curves, all seeds

Usage: python -m src.plots
"""
import glob
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config

FIVE = ["FCFS", "Round Robin", "Min-Min", "Energy-focused DQN", "Multi-objective DQN"]
COLORS = {"FCFS": "#9aa5b1", "Round Robin": "#c3a37a", "Min-Min": "#7f8fa6",
          "Energy-focused DQN": "#2e8b57", "Multi-objective DQN": "#1f5fa8"}
METRICS = [("energy_kwh", "Energy consumption (kWh)", "energy_comparison.png"),
           ("makespan_s", "Makespan (s)", "makespan_comparison.png"),
           ("active_utilization", "Active CPU utilization (awake servers)", "utilization_comparison.png"),
           ("load_imbalance", "Load imbalance (std of server utilization)", "load_balance_comparison.png"),
           ("sla_violation_pct", "SLA violations (% of tasks)", "sla_violations_comparison.png")]
SCENARIOS = ["light", "medium", "heavy", "burst"]


def grouped_bars(df, metric, ylabel, fname, groups, group_col, title):
    fig, ax = plt.subplots(figsize=(9, 4.6))
    width = 0.8 / len(FIVE)
    x = np.arange(len(groups))
    for i, s in enumerate(FIVE):
        sub = df[df["scheduler"] == s].set_index(group_col).reindex(groups)
        ax.bar(x + (i - 2) * width, sub[f"{metric}_mean"], width, yerr=sub[f"{metric}_sd"],
               capsize=2, label=s, color=COLORS[s], edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([str(g).capitalize() if isinstance(g, str) else f"{g} servers" for g in groups])
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / fname, dpi=160)
    plt.close(fig)


def curves():
    logs = {}
    for p in sorted(glob.glob(str(config.RESULTS_DIR / "training_log_*_seed*.csv"))):
        m = re.search(r"training_log_(.+)_seed(\d+)\.csv", p)
        logs.setdefault(m.group(1), []).append(pd.read_csv(p))
    if not logs:
        return
    labels = {"energy_dqn": "Energy-focused DQN", "multi_dqn_balanced": "Multi-objective DQN"}
    for col, ylabel, fname, smooth in [("reward", "Mean per-step reward", "training_reward.png", 10),
                                       ("loss", "Huber loss", "training_loss.png", 10),
                                       ("val_score", "Score vs FCFS (lower = better)",
                                        "validation_score.png", 1)]:
        fig, axes = plt.subplots(1, len(logs), figsize=(5.2 * len(logs), 3.6), squeeze=False)
        for ax, (name, runs) in zip(axes[0], logs.items()):
            for k, r in enumerate(runs):
                s = r[["episode", col]].dropna()
                y = s[col].rolling(smooth, min_periods=1).mean() if smooth > 1 else s[col]
                ax.plot(s["episode"], y, lw=1.4, alpha=0.85, label=f"seed {k}")
            if col == "val_score":
                ax.axhline(1.0, color="grey", ls="--", lw=1, label="FCFS")
                js = sorted(glob.glob(str(config.RESULTS_DIR / f"training_summary_{name}_seed*.json")))
                if js:
                    import json
                    ax.axhline(json.load(open(js[0]))["random_val_score"], color="#c0504d", ls=":", lw=1,
                               label="Random")
            ax.set_title(labels.get(name, name), loc="left", fontsize=11)
            ax.set_xlabel("Episode")
            ax.set_ylabel(ylabel)
            ax.spines[["top", "right"]].set_visible(False)
            ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(config.FIGURES_DIR / fname, dpi=160)
        plt.close(fig)


def main(summary_file="summary.csv"):
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    s = pd.read_csv(config.RESULTS_DIR / summary_file)
    main_df = s[s["servers"] == config.TEST_SERVERS]
    scen = [x for x in SCENARIOS if x in set(main_df["scenario"])]
    for metric, ylabel, fname in METRICS:
        grouped_bars(main_df, metric, ylabel, fname, scen, "scenario",
                     f"{ylabel.split(' (')[0]} by workload scenario ({config.TEST_SERVERS} servers, mean ± SD)")
    gen = s[s["scenario"] == "medium"]
    sizes = sorted(gen["servers"].unique())
    if len(sizes) > 1:
        grouped_bars(gen, "energy_kwh", "Energy consumption (kWh)", "generalization_energy.png",
                     sizes, "servers", "Energy at medium load across cluster sizes (same trained models)")
        grouped_bars(gen, "makespan_s", "Makespan (s)", "generalization_makespan.png",
                     sizes, "servers", "Makespan at medium load across cluster sizes (same trained models)")
    curves()
    print("figures written to", config.FIGURES_DIR)


if __name__ == "__main__":
    main()
