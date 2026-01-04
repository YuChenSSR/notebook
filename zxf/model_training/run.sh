set -e
set -x

### 参数
project_dir="/home/idc2/notebook/zxf/model_training"
data_dir="/home/idc2/notebook/zxf/data/modoel_training"
market_name="csi800"
seed_num=5
today=$(date +%Y%m%d)

### 激活虚拟环境
cd "$project_dir"
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate pytorch

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
# folder_name="${market_name}_${today}_${start_date}_${end_date}"
# expt_path="${data_dir}/${folder_name}"
# mkdir -p "$expt_path"
# echo "Exp_path:$expt_path"
# cp ${data_dir}/workflow_config_master_Alpha158_${market_name}.yaml ${expt_path}/


# ### 主实验
cd "$project_dir"
# python3 main.py --market_name="$market_name" --folder_name="$folder_name" --seed_num="$seed_num" --data_path="$data_dir"

### 生成预测值
folder_name="csi800_20251211_20150101_20251208"
# python3 processing_pred.py --data_path="$data_dir" --market_name="$market_name" --folder_name="$folder_name"

# ### qlib回测
# python3 Qlib_backtest.py --market_name="$market_name" --folder_name="$folder_name" --data_path="$data_dir"

# ### my回测
python3 run_my_backtest.py --market_name="$market_name" --folder_name="$folder_name" --data_path="$data_dir"
