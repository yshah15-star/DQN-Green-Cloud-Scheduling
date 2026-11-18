"""
Clean the Alibaba batch_task data and split it into train / val / test.

Steps:
  1. Keep finished tasks with valid times, CPU and memory
  2. Keep tasks lasting 10 s to 1 hour and needing at most 16 cores
  3. Take a random sample of 200,000 tasks, sorted by time
  4. Add deadline = arrival + 2 x duration
  5. Split by time: first 70% train, next 15% val, last 15% test

Run: python -m src.preprocess_trace --input data/raw/batch_task_sample.csv
"""

import argparse


import numpy as np
import pandas as pd
import config


def read_trace(path):
    """Read the Alibaba batch_task file (.csv or .tar.gz) in chunks of 1 million rows."""
    return pd.read_csv(path, header=None, names=config.BATCH_TASK_COLUMNS,
                       chunksize=1_000_000, low_memory=False)


def main(input_path, seed=42):
    rng = np.random.default_rng(seed)
    
    kept = []
    counts = dict(rows=0, terminated=0, valid_time=0, duration_ok=0, cpu_ok=0)
    for chunk in read_trace(input_path):
        counts["rows"] += len(chunk)
        c = chunk[chunk["status"] == "Terminated"]
        counts["terminated"] += len(c)
        c = c[(c["start_time"] > 0) & (c["end_time"] > c["start_time"])
              & (c["plan_cpu"] > 0) & (c["plan_mem"] > 0)]
        counts["valid_time"] += len(c)
        d = c["end_time"] - c["start_time"]
        c = c[(d >= config.MIN_DURATION_S) & (d <= config.MAX_DURATION_S)]
        counts["duration_ok"] += len(c)
        c = c[c["plan_cpu"] / 100.0 <= config.MAX_TASK_CORES]
        counts["cpu_ok"] += len(c)
        kept.append(c)
        
    df = pd.concat(kept, ignore_index=True)
    if len(df) > config.SAMPLE_ROWS:
        df = df.iloc[np.sort(rng.choice(len(df), config.SAMPLE_ROWS, replace=False))]
    df = df.sort_values("start_time", kind="stable").reset_index(drop=True)

    config.DATA_RAW.mkdir(parents=True, exist_ok=True)
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    # same headerless format as the original trace, so the sample can be re-processed identically
    df.to_csv(config.DATA_RAW / "batch_task_sample.csv", index=False, header=False)

    out = pd.DataFrame({
        "task_id": np.arange(len(df)),
        "arrival_time": (df["start_time"] - df["start_time"].min()).astype(float),
        "duration": (df["end_time"] - df["start_time"]).astype(float),
        "cpu_request": df["plan_cpu"] / 100.0,
        "memory_request": df["plan_mem"].astype(float),
        "job_name": df["job_name"].values,
        "task_name": df["task_name"].values,
    })
    out["deadline"] = out["arrival_time"] + out["duration"] * config.SLACK_FACTOR

    n = len(out)                      # 200,000 tasks, already sorted by time
    train_end = int(n * 0.70)         # 140,000
    val_end = int(n * 0.85)           # 170,000

    train = out.iloc[0:train_end]         # earliest 70%
    val = out.iloc[train_end:val_end]     # next 15%
    test = out.iloc[val_end:n]            # latest 15%

    # ---------- Part 6: save the three files ----------
    train.to_csv(config.DATA_PROCESSED / "train.csv", index=False)
    val.to_csv(config.DATA_PROCESSED / "val.csv", index=False)
    test.to_csv(config.DATA_PROCESSED / "test.csv", index=False)

    print("Rows after each filter:", counts)
    print("Train:", len(train), "tasks")
    print("Val:  ", len(val), "tasks")
    print("Test: ", len(test), "tasks")
    


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(args.input, args.seed)
