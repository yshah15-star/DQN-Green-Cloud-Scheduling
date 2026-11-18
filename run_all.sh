#!/usr/bin/env bash
# Full pipeline: preprocess -> hyper-parameter search -> train 5 seeds x 2 models -> experiments -> figures.
# Usage: bash run_all.sh path/to/batch_task.tar.gz      (about 30-40 minutes on a laptop CPU)
set -euo pipefail
INPUT=${1:-data/raw/batch_task_sample.csv}

python -m src.preprocess_trace --input "$INPUT"
python -m src.train --reward energy --search --episodes 200
python -m src.train --reward multi  --search --episodes 200
for s in 0 1 2 3 4; do
  python -m src.train --reward energy --seed "$s"
  python -m src.train --reward multi --preset balanced --seed "$s"
done
python -m src.run_experiments
python -m src.plots
