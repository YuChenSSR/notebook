#!/bin/bash
set -euo pipefail
set -x

# ===== Env =====
source /opt/anaconda3/etc/profile.d/conda.sh
export CONDA_NO_PLUGINS=true
conda activate pytorch

REPO="/home/idc2/notebook"
FUTURES_DATA="${REPO}/futures/data"
CODE_DIR="${REPO}/futures/code/dump_qlib"
QLIB_SCRIPTS="${REPO}/qlib/scripts"

# ===== Tunables =====
WORKERS_AGG="${WORKERS_AGG:-8}"          # 聚合并发
CHUNKSIZE="${CHUNKSIZE:-500000}"         # 聚合读csv的chunk大小
WORKERS_NORM="${WORKERS_NORM:-8}"        # 标准化并发
OVERWRITE="${OVERWRITE:-1}"              # 1=覆盖重跑；0=已有文件则跳过
OVERWRITE_CALENDAR="${OVERWRITE_CALENDAR:-1}"  # 1=重建交易日历；0=复用已有
CLEAN="${CLEAN:-0}"                      # 1=清空派生目录再跑
SANITY="${SANITY:-1}"                    # 1=跑 Step D 校验；0=跳过
# Step D 默认使用主力连续（*88）配置；如需全市场可通过环境变量覆盖 SANITY_CONFIG
SANITY_CONFIG="${SANITY_CONFIG:-${REPO}/futures/workflow_config_futures_Alpha158_f88.yaml}"  # Step D 用的 config

# ===== Safety checks =====
if [[ ! -d "${FUTURES_DATA}/raw_data/futures_klines_1m" ]]; then
  echo "ERROR: 1m raw data dir not found: ${FUTURES_DATA}/raw_data/futures_klines_1m"
  exit 1
fi

if [[ "${CLEAN}" == "1" ]]; then
  echo "=== CLEAN: Removing derived directories ==="
  rm -rf "${FUTURES_DATA}/qlib_data" "${FUTURES_DATA}/qlib_bin"
fi
mkdir -p "${FUTURES_DATA}/qlib_data" "${FUTURES_DATA}/qlib_bin"

echo "=== Step A: 1m -> 1d 聚合生成 qlib-source (raw_data + backtest_source) === $(date)"
python "${CODE_DIR}/dump_futures_qlib_source.py" \
  --data_path "${FUTURES_DATA}" \
  --in_1m_dir "${FUTURES_DATA}/raw_data/futures_klines_1m" \
  --max_workers "${WORKERS_AGG}" \
  --chunksize "${CHUNKSIZE}" \
  --overwrite "${OVERWRITE}"

echo "=== Step B: 标准化生成 qlib_train_source + trade_calendar.csv === $(date)"
python "${CODE_DIR}/normalize_futures.py" \
  --data_path "${FUTURES_DATA}" \
  --max_workers "${WORKERS_NORM}" \
  --overwrite_calendar "${OVERWRITE_CALENDAR}" \
  --overwrite "${OVERWRITE}"

echo "=== Step C1: dump_bin -> cn_data_train === $(date)"
python "${QLIB_SCRIPTS}/dump_bin.py" dump_all \
  --csv_path "${FUTURES_DATA}/qlib_data/qlib_train_source" \
  --qlib_dir "${FUTURES_DATA}/qlib_bin/cn_data_train" \
  --date_field_name=tradedate \
  --exclude_fields=tradedate,symbol

echo "=== Step C2: dump_bin -> cn_data_backtest === $(date)"
python "${QLIB_SCRIPTS}/dump_bin.py" dump_all \
  --csv_path "${FUTURES_DATA}/qlib_data/qlib_backtest_source" \
  --qlib_dir "${FUTURES_DATA}/qlib_bin/cn_data_backtest" \
  --date_field_name=tradedate \
  --exclude_fields=tradedate,symbol

echo "=== Step C3: 生成 instruments/f88.txt（主力连续 *88） === $(date)"
python "${CODE_DIR}/build_instruments_f88.py" \
  --qlib_dir "${FUTURES_DATA}/qlib_bin/cn_data_train" \
  --overwrite "${OVERWRITE}"
python "${CODE_DIR}/build_instruments_f88.py" \
  --qlib_dir "${FUTURES_DATA}/qlib_bin/cn_data_backtest" \
  --overwrite "${OVERWRITE}"

if [[ "${SANITY}" == "1" ]]; then
  echo "=== Step D: handler/dataset 快速验证（可选） === $(date)"
  echo "SANITY_CONFIG=${SANITY_CONFIG}"
  python "${REPO}/futures/code/data_generator_futures.py" --config "${SANITY_CONFIG}" --overwrite "${OVERWRITE}"
fi

echo "=== ALL DONE === $(date)"

