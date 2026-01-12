#!/bin/bash
set -euo pipefail

# master_incremental_oneclick.sh
#
# 增量训练一键脚本：
# - INCREMENTAL=0（默认）：保持现状，从头训练
# - INCREMENTAL=1：从 PREV_FOLDER_NAME 指定的上一轮实验目录 warm-start，并把窗口滚动到最新交易日

# ===== 参数（可通过环境变量覆盖）=====
PROJECT_DIR="${PROJECT_DIR:-/home/idc2/notebook/zxf/code}"
DATA_DIR="${DATA_DIR:-/home/idc2/notebook/zxf/data}"
MARKET_NAME="${MARKET_NAME:-csi800}"
SEED_NUM="${SEED_NUM:-5}"
QLIB_PATH="${QLIB_PATH:-/home/idc2/notebook/qlib_bin/cn_data_train}"
N_EPOCHS_OVERRIDE="${N_EPOCHS_OVERRIDE:-}"

INCREMENTAL="${INCREMENTAL:-0}"
PREV_FOLDER_NAME="${PREV_FOLDER_NAME:-}"

# 默认新实验目录名：{market}_{today}_inc_{prev}
TODAY="$(date +%Y%m%d)"
FOLDER_NAME="${FOLDER_NAME:-${MARKET_NAME}_${TODAY}_inc}"

# ===== Python env =====
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate pytorch
PYTHON="${CONDA_PREFIX}/bin/python"

cd "${PROJECT_DIR}/Master"

if [[ "${INCREMENTAL}" == "1" ]]; then
  if [[ -z "${PREV_FOLDER_NAME}" ]]; then
    echo "ERROR: INCREMENTAL=1 需要设置 PREV_FOLDER_NAME（上一轮实验目录名）"
    exit 1
  fi
  "${PYTHON}" main.py \
    --market_name="${MARKET_NAME}" \
    --folder_name="${FOLDER_NAME}" \
    --seed_num="${SEED_NUM}" \
    --data_path="${DATA_DIR}" \
    --incremental=True \
    --prev_folder_name="${PREV_FOLDER_NAME}" \
    --qlib_path="${QLIB_PATH}" \
    --roll_to_latest=True \
    --resume_from=best \
    --best_metric=valid_IC \
    ${N_EPOCHS_OVERRIDE:+--n_epochs_override=${N_EPOCHS_OVERRIDE}}
else
  "${PYTHON}" main.py \
    --market_name="${MARKET_NAME}" \
    --folder_name="${FOLDER_NAME}" \
    --seed_num="${SEED_NUM}" \
    --data_path="${DATA_DIR}" \
    --incremental=False \
    ${N_EPOCHS_OVERRIDE:+--n_epochs_override=${N_EPOCHS_OVERRIDE}}
fi

