set -e
# 默认不打印 bash 逐行执行细节；需要排查问题时可用：MASTER_DEBUG=1 bash master_oneclick.sh
if [ "${MASTER_DEBUG:-0}" = "1" ]; then
  set -x
fi

### 参数
project_dir="/home/idc2/notebook/zxf/code"
data_dir="/home/idc2/notebook/zxf/data"
market_name="csi800"
seed_num=5
today=$(date +%Y%m%d)

### 激活虚拟环境
cd "$project_dir"
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate pytorch

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
cd "$project_dir"
read -r start_date end_date <<< $(python - <<EOF
import yaml
from datetime import datetime

with open("${data_dir}/workflow_config_master_Alpha158_${market_name}.yaml", 'r') as f:
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
folder_name="${market_name}_${today}_${start_date}_${end_date}"
# folder_name="csi800_20260114_20150101_20260113"
echo "Folder_name:$folder_name"

### 数据切割
cd "$project_dir"
python3 data_generator.py --market_name="$market_name" --folder_name="$folder_name"
cp ${data_dir}/workflow_config_master_Alpha158_${market_name}.yaml ${data_dir}/master_results/${folder_name}/

### 主实验
cd "$project_dir/Master"
"$PYTHON" main.py --market_name="$market_name" --folder_name="$folder_name" --seed_num="$seed_num"

### 生成预测值
cd "$project_dir/Backtest"
"$PYTHON" processing_prediction_seed_step.py --data_path="$data_dir" --market_name="$market_name" --folder_name="$folder_name"

### 回测
cd "$project_dir/Backtest"
"$PYTHON" backtest_seed_step.py --market_name="$market_name" --folder_name="$folder_name"
 
