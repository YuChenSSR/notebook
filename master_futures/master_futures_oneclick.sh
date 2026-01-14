#!/bin/bash
set -euo pipefail

# master_futures_oneclick.sh
#
# 期货版 MASTER 一键脚本（物理隔离：代码在 master_futures/，产物在 master_futures/data/）
#
# 两种模式：
# - INCREMENTAL=0（默认）：从 base config 生成数据并从头训练
# - INCREMENTAL=1：从 PREV_FOLDER_NAME 指定的上一轮实验 warm-start（会滚动窗口到最新交易日）
#
# 依赖：
# - futures 数据层 qlib_bin 已准备（默认 provider_uri 指向 futures/data/qlib_bin/cn_data_train）
# - instruments/f88.txt 已存在（由 futures/run_futures_full_pipeline.sh 生成）

REPO="${REPO:-/home/idc2/notebook}"
MF_DIR="${MF_DIR:-${REPO}/master_futures}"
CODE_DIR="${CODE_DIR:-${MF_DIR}/code}"
DATA_DIR="${DATA_DIR:-${MF_DIR}/data}"

MARKET_NAME="${MARKET_NAME:-f88}"
QLIB_PATH="${QLIB_PATH:-${REPO}/futures/data/qlib_bin/cn_data_train}"

INCREMENTAL="${INCREMENTAL:-0}"
PREV_FOLDER_NAME="${PREV_FOLDER_NAME:-}"

SEED_NUM="${SEED_NUM:-5}"
N_EPOCHS_OVERRIDE="${N_EPOCHS_OVERRIDE:-}"

TODAY="$(date +%Y%m%d)"
FOLDER_NAME="${FOLDER_NAME:-${MARKET_NAME}_${TODAY}_exp}"

BASE_CONFIG="${BASE_CONFIG:-${DATA_DIR}/workflow_config_master_Alpha158_${MARKET_NAME}.yaml}"

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate pytorch
PYTHON="${CONDA_PREFIX}/bin/python"

if [[ ! -f "${QLIB_PATH}/instruments/f88.txt" ]]; then
  echo "ERROR: 找不到 f88.txt: ${QLIB_PATH}/instruments/f88.txt"
  echo "请先运行：${REPO}/futures/run_futures_full_pipeline.sh"
  exit 1
fi

if [[ "${INCREMENTAL}" == "1" ]]; then
  if [[ -z "${PREV_FOLDER_NAME}" ]]; then
    echo "ERROR: INCREMENTAL=1 需要设置 PREV_FOLDER_NAME（上一轮实验目录名，位于 ${DATA_DIR}/master_results/ 下）"
    exit 1
  fi
  cd "${CODE_DIR}/Master"
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
  # baseline：生成 dl_* + 从头训练
  "${PYTHON}" "${CODE_DIR}/data_generator.py" \
    --market_name="${MARKET_NAME}" \
    --qlib_path="${QLIB_PATH}" \
    --data_path="${DATA_DIR}" \
    --folder_name="${FOLDER_NAME}" \
    --config_path="${BASE_CONFIG}" \
    --overwrite_exp_config=True

  cd "${CODE_DIR}/Master"
  "${PYTHON}" main.py \
    --market_name="${MARKET_NAME}" \
    --folder_name="${FOLDER_NAME}" \
    --seed_num="${SEED_NUM}" \
    --data_path="${DATA_DIR}" \
    --incremental=False \
    ${N_EPOCHS_OVERRIDE:+--n_epochs_override=${N_EPOCHS_OVERRIDE}}
fi

