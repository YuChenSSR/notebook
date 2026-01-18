#!/usr/bin/env bash
# set -euo pipefail
set -e  # 只保留这个，去掉 pipefail
set -u  # 保留这个

# source /opt/anaconda3/etc/profile.d/conda.sh
# conda activate pytorch

# PYTHON="${CONDA_PREFIX}/bin/python"
# export PATH="${CONDA_PREFIX}/bin:${PATH}"
mkdir -p ~/notebook/zxf/data/Daily_data/Good_seed/seed3

mkdir -p ~/notebook/zxf/data/Daily_data/Good_seed/seed3/backtest_output



/opt/anaconda3/bin/conda run -n pytorch python ~/notebook/zxf/code/Backtest/My_backtest_auto_in_out.py

