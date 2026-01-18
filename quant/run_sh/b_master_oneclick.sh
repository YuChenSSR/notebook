set -e
set -x

### 参数
project_dir="/home/idc2/notebook/zxf/quant"
data_dir="/home/idc2/notebook/zxf/quant/data"
market_name="csi800"
folder_name="csi800_128_20150101_20260116"
seed_num=6

data_path="$data_dir/experimental_results/$folder_name"

# ### 激活虚拟环境
cd "$project_dir"
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate pytorch

### 读取workflow_config参数
cd "$project_dir"
read -r start_date end_date <<< $(python3 - <<EOF
import yaml
from datetime import datetime

with open("${data_path}/workflow_config_master_Alpha158_${market_name}.yaml", 'r') as f:
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


### 数据切割
cd "$project_dir/feature_processing"
python3 data_generator.py --market_name="$market_name"  --data_path="$data_path"

### 主实验
cd "$project_dir/master"
python3 main.py --market_name="$market_name" --seed_num="$seed_num" --data_path="$data_path"

### 生成预测值
cd "$project_dir/backtest"
python3 prediction_master_to_pred.py --data_path="$data_path" --market_name="$market_name" --is_batch=True

### 回测
cd "$project_dir/backtest"
python3 qlib_backtest.py --market_name="$market_name" --data_path="$data_path" --is_batch=True