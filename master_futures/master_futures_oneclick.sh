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
# 兼容旧变量名：QLIB_PATH（训练数据）
QLIB_TRAIN_PATH="${QLIB_TRAIN_PATH:-${QLIB_PATH:-${REPO}/futures/data/qlib_bin/cn_data_train}}"
QLIB_BACKTEST_PATH="${QLIB_BACKTEST_PATH:-${REPO}/futures/data/qlib_bin/cn_data_backtest}"

INCREMENTAL="${INCREMENTAL:-0}"
PREV_FOLDER_NAME="${PREV_FOLDER_NAME:-}"

SEED_NUM="${SEED_NUM:-5}"
N_EPOCHS_OVERRIDE="${N_EPOCHS_OVERRIDE:-}"

# ============================
# 四个布尔值开关（你只需要关心这四个）
# - GEN_DATA: 数据切割/生成三个 pkl（dl_train/dl_valid/dl_test）
# - TRAIN   : 训练主实验（生成 ckpt）
# - PREDICT : 生成预测值（由 ckpt + dl_test 生成 prediction csv）
# - BACKTEST: 回测（读取 predictions 目录，Qlib 仿真回测）
# 取值支持：True/False 或 1/0（大小写不敏感）
# ============================
GEN_DATA="${GEN_DATA:-True}"
TRAIN="${TRAIN:-True}"
PREDICT="${PREDICT:-True}"
BACKTEST="${BACKTEST:-True}"

TODAY="$(date +%Y%m%d)"
FOLDER_NAME="${FOLDER_NAME:-${MARKET_NAME}_${TODAY}_exp}"

BASE_CONFIG="${BASE_CONFIG:-${DATA_DIR}/workflow_config_master_Alpha158_${MARKET_NAME}.yaml}"

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate pytorch
PYTHON="${CONDA_PREFIX}/bin/python"

is_true() {
  local v
  v="$(echo "${1:-}" | tr '[:upper:]' '[:lower:]')"
  case "$v" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

### 吞吐/稳定性：默认走安全配置；如需提速可自行设 MASTER_NUM_WORKERS>0
export MASTER_NUM_WORKERS="${MASTER_NUM_WORKERS:-0}"
export MASTER_PIN_MEMORY="${MASTER_PIN_MEMORY:-1}"
export MASTER_PERSISTENT_WORKERS="${MASTER_PERSISTENT_WORKERS:-1}"
export MASTER_PREFETCH_FACTOR="${MASTER_PREFETCH_FACTOR:-2}"
export MASTER_NON_BLOCKING="${MASTER_NON_BLOCKING:-1}"
export MASTER_EVAL_FREQ="${MASTER_EVAL_FREQ:-0}"   # 训练时默认不做每 epoch valid 评估（更快）；如需可改成 1

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

if [[ ! -f "${QLIB_TRAIN_PATH}/instruments/f88.txt" ]]; then
  echo "ERROR: 找不到 f88.txt: ${QLIB_TRAIN_PATH}/instruments/f88.txt"
  echo "请先运行：${REPO}/futures/run_futures_full_pipeline.sh"
  exit 1
fi

if [[ "${BACKTEST}" == "1" ]]; then
  if [[ ! -f "${QLIB_BACKTEST_PATH}/instruments/f88.txt" ]]; then
    echo "ERROR: 找不到 f88.txt: ${QLIB_BACKTEST_PATH}/instruments/f88.txt"
    echo "请先运行：${REPO}/futures/run_futures_full_pipeline.sh"
    exit 1
  fi
fi

EXP_DIR="${DATA_DIR}/master_results/${FOLDER_NAME}"
CKPT_DIR="${EXP_DIR}/Master_results"
PRED_OUT_DIR="${EXP_DIR}/Backtest_Results/predictions"
BT_OUT="${EXP_DIR}/Backtest_Results/results/backtest_result.csv"

### Step 1) 数据切割生成三个 pkl 数据集
if is_true "${GEN_DATA}"; then
  "${PYTHON}" "${CODE_DIR}/data_generator.py" \
    --market_name="${MARKET_NAME}" \
    --qlib_path="${QLIB_TRAIN_PATH}" \
    --data_path="${DATA_DIR}" \
    --folder_name="${FOLDER_NAME}" \
    --config_path="${BASE_CONFIG}" \
    --overwrite_exp_config=True
fi

### Step 2) 训练主实验（生成 ckpt）
if is_true "${TRAIN}"; then
  if [[ ! -f "${EXP_DIR}/${MARKET_NAME}_self_dl_train.pkl" ]] || [[ ! -f "${EXP_DIR}/${MARKET_NAME}_self_dl_valid.pkl" ]] || [[ ! -f "${EXP_DIR}/${MARKET_NAME}_self_dl_test.pkl" ]]; then
    echo "ERROR: 找不到切分后的三个数据集（dl_train/dl_valid/dl_test）。"
    echo "请先设置 GEN_DATA=True，或确保文件存在于：${EXP_DIR}"
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
      --qlib_path="${QLIB_TRAIN_PATH}" \
    --roll_to_latest=True \
    --resume_from=best \
    --best_metric=valid_IC \
    ${N_EPOCHS_OVERRIDE:+--n_epochs_override=${N_EPOCHS_OVERRIDE}}
else
  cd "${CODE_DIR}/Master"
  "${PYTHON}" main.py \
    --market_name="${MARKET_NAME}" \
    --folder_name="${FOLDER_NAME}" \
    --seed_num="${SEED_NUM}" \
    --data_path="${DATA_DIR}" \
    --incremental=False \
    ${N_EPOCHS_OVERRIDE:+--n_epochs_override=${N_EPOCHS_OVERRIDE}}
  fi
fi

### 生成预测值（对齐股票版流程：Backtest/processing_prediction_seed_step.py）
if is_true "${PREDICT}"; then
  mkdir -p "${PRED_OUT_DIR}"
  if ! compgen -G "${CKPT_DIR}/*self_exp*.pkl" > /dev/null; then
    echo "ERROR: 找不到训练 ckpt：${CKPT_DIR}/*self_exp*.pkl"
    echo "请先设置 TRAIN=True 完成训练。"
    exit 1
  fi
  cd "${CODE_DIR}/Backtest"
  "${PYTHON}" "${CODE_DIR}/Backtest/processing_prediction_seed_step.py" \
    --data_path="${DATA_DIR}" \
    --market_name="${MARKET_NAME}" \
    --folder_name="${FOLDER_NAME}" \
    --overwrite=1
fi

if is_true "${BACKTEST}"; then
  if ! compgen -G "${PRED_OUT_DIR}/master_predictions_backday_*.csv" > /dev/null; then
    echo "ERROR: 找不到预测文件：${PRED_OUT_DIR}/master_predictions_backday_*.csv"
    echo "请先设置 PREDICT=True 生成预测值。"
    exit 1
  fi
  cd "${CODE_DIR}/Backtest"
  "${PYTHON}" "${CODE_DIR}/Backtest/backtest_seed_step.py" \
    --market_name="${MARKET_NAME}" \
    --folder_name="${FOLDER_NAME}" \
    --data_path="${DATA_DIR}" \
    --qlib_path="${QLIB_BACKTEST_PATH}"
fi

