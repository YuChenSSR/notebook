#!/bin/bash
set -euo pipefail

# 默认不打印 bash 逐行执行细节；需要排查问题时可用：MASTER_DEBUG=1 bash master_oneclick.sh
if [ "${MASTER_DEBUG:-0}" = "1" ]; then
  set -x
fi

### 路径与参数（均可用环境变量覆盖）
# 让脚本在任意工作目录都能运行：默认以脚本所在目录作为工程目录（即 .../zxf/code）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-${SCRIPT_DIR}}"
DATA_DIR="${DATA_DIR:-$(cd "${PROJECT_DIR}/../data" && pwd)}"

MARKET_NAME="${MARKET_NAME:-csi800}"
SEED_NUM="${SEED_NUM:-5}"

# Qlib 二进制数据路径（训练与回测可不同；按需覆盖）
QLIB_TRAIN_PATH="${QLIB_TRAIN_PATH:-/home/idc2/notebook/qlib_bin/cn_data_train}"
QLIB_BACKTEST_PATH="${QLIB_BACKTEST_PATH:-/home/idc2/notebook/qlib_bin/cn_data_backtest}"

# 是否自动生成（切割）实验数据：当 master_results/{FOLDER_NAME} 缺少必要 pkl 时会触发
AUTO_GENERATE_DATA="${AUTO_GENERATE_DATA:-1}"

TODAY="$(date +%Y%m%d)"

### 激活虚拟环境
cd "${PROJECT_DIR}"
CONDA_SH="${CONDA_SH:-/opt/anaconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-pytorch}"
if [ ! -f "${CONDA_SH}" ]; then
  echo "ERROR: 找不到 conda 初始化脚本: ${CONDA_SH}"
  echo "请设置环境变量 CONDA_SH=/path/to/conda.sh"
  exit 1
fi
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

# 强制使用当前 conda 环境的 python，避免被 PATH 中其它 python 覆盖
PYTHON="${CONDA_PREFIX}/bin/python"
export PATH="${CONDA_PREFIX}/bin:${PATH}"

### GPU 吞吐优化（不改变数值结果；仅提升数据搬运/预取与减少同步点）
# 如遇到 DataLoader 多进程不兼容，可临时回退：export MASTER_NUM_WORKERS=0
export MASTER_NUM_WORKERS="${MASTER_NUM_WORKERS:-4}"
export MASTER_PIN_MEMORY="${MASTER_PIN_MEMORY:-1}"
export MASTER_PERSISTENT_WORKERS="${MASTER_PERSISTENT_WORKERS:-1}"
export MASTER_PREFETCH_FACTOR="${MASTER_PREFETCH_FACTOR:-2}"
export MASTER_NON_BLOCKING="${MASTER_NON_BLOCKING:-1}"

# 防止 CPU 线程过度争用（尤其是 num_workers>0 时），通常有助于稳定提速
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

### 读取workflow_config参数
cd "${PROJECT_DIR}"
read -r start_date end_date <<< $("${PYTHON}" - <<EOF
import yaml
from datetime import datetime

with open("${DATA_DIR}/workflow_config_master_Alpha158_${MARKET_NAME}.yaml", 'r') as f:
    config = yaml.safe_load(f)

start_date = config["task"]["dataset"]["kwargs"]["segments"]["train"][0].strftime("%Y%m%d")
end_date = config["task"]["dataset"]["kwargs"]["segments"]["test"][1].strftime("%Y%m%d")
print(f"{start_date} {end_date}")
EOF
)
if [ -z "$start_date" ] || [ -z "$end_date" ]; then
    echo "Error: Python script failed to extract dates."
    exit 1
fi


### 生成实验主目录
# 默认：{market}_{today}_{train_start}_{test_end}；可通过环境变量 FOLDER_NAME 覆盖
FOLDER_NAME="${FOLDER_NAME:-${MARKET_NAME}_${TODAY}_${start_date}_${end_date}}"
echo "Folder_name:${FOLDER_NAME}"

### 数据切割/实验数据生成（必要文件缺失时自动生成）
EXP_DIR="${DATA_DIR}/master_results/${FOLDER_NAME}"
REQ_1="${EXP_DIR}/workflow_config_master_Alpha158_${MARKET_NAME}.yaml"
REQ_2="${EXP_DIR}/${MARKET_NAME}_self_dl_train.pkl"
REQ_3="${EXP_DIR}/${MARKET_NAME}_self_dl_valid.pkl"
REQ_4="${EXP_DIR}/${MARKET_NAME}_self_dl_test.pkl"

if [ "${AUTO_GENERATE_DATA}" = "1" ]; then
  if [ ! -f "${REQ_1}" ] || [ ! -f "${REQ_2}" ] || [ ! -f "${REQ_3}" ] || [ ! -f "${REQ_4}" ]; then
    echo "[INFO] 实验数据缺失，开始生成：${EXP_DIR}"
    "${PYTHON}" "${PROJECT_DIR}/data_generator.py" \
      --market_name="${MARKET_NAME}" \
      --qlib_path="${QLIB_TRAIN_PATH}" \
      --data_path="${DATA_DIR}" \
      --folder_name="${FOLDER_NAME}"
  else
    echo "[INFO] 已检测到实验数据齐全，跳过 data_generator"
  fi
fi

### 主实验
cd "${PROJECT_DIR}/Master"
"${PYTHON}" main.py \
  --market_name="${MARKET_NAME}" \
  --folder_name="${FOLDER_NAME}" \
  --seed_num="${SEED_NUM}" \
  --data_path="${DATA_DIR}" \
  --qlib_path="${QLIB_TRAIN_PATH}"

### 生成预测值
cd "${PROJECT_DIR}/Backtest"
"${PYTHON}" processing_prediction_seed_step.py \
  --data_path="${DATA_DIR}" \
  --market_name="${MARKET_NAME}" \
  --folder_name="${FOLDER_NAME}"

### 回测
cd "${PROJECT_DIR}/Backtest"
"${PYTHON}" backtest_seed_step.py \
  --market_name="${MARKET_NAME}" \
  --folder_name="${FOLDER_NAME}" \
  --data_path="${DATA_DIR}" \
  --qlib_path="${QLIB_BACKTEST_PATH}"
 
