set -e
set -x

### 参数
project_dir="/home/a/notebook/zxf/code"
data_dir="/home/a/notebook/zxf/data"
market_name="csi800"
d_model=128
seed_num=6
today=$(date +%Y%m%d)

# ### 激活虚拟环境
# cd "$project_dir"
# source /opt/anaconda3/etc/profile.d/conda.sh
# conda activate pytorch

### 读取workflow_config参数
cd "$project_dir"
read -r start_date end_date <<< $(python3 - <<EOF
import yaml
from datetime import datetime

with open("${data_dir}/qlib/workflow_config/workflow_config_master_Alpha158_${market_name}_${d_model}.yaml", 'r') as f:
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
# folder_name="${market_name}_${today}_${start_date}_${end_date}"
folder_name="csi800_128_20260117_20250101_20260116"
echo "Folder_name:$folder_name"
data_path="$data_dir/master_results"

### 数据切割
# cd "$project_dir/data_generator"
# python3 data_generator.py --market_name="$market_name" --folder_name="$folder_name" --data_path="$data_path" --d_model="$d_model"
# cp ${data_dir}/qlib/workflow_config/workflow_config_master_Alpha158_${market_name}_${d_model}.yaml ${data_path}/${folder_name}/

### 主实验
cd "$project_dir/master"
python3 main.py --market_name="$market_name" --folder_name="$folder_name" --seed_num="$seed_num" --data_path="$data_path" --d_model="$d_model"

### 生成预测值
cd "$project_dir/daily_training"
python3 processing_prediction.py --data_path="$data_path" --market_name="$market_name" --folder_name="$folder_name" --d_model="$d_model"

### 回测
cd "$project_dir/Backtest"
python3 backtest_seed_step.py --market_name="$market_name" --folder_name="$folder_name"