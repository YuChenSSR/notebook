set -e
set -x

### 参数
project_dir="/home/idc2/notebook/quant"
data_dir="/home/idc2/notebook/quant/data"
market_name="csi800"
folder_name="csi800_771_20260121_20150101_20260116"

seed_num=3


data_path="$data_dir/experimental_results/$folder_name"

# ### 激活虚拟环境
cd "$project_dir"
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate pytorch


### 数据切割
# cd "$project_dir/feature_processing"
# python3 data_generator.py --market_name="$market_name"  --data_path="$data_path"

### 主实验
cd "$project_dir/master"
python3 main.py --market_name="$market_name" --seed_num="$seed_num" --data_path="$data_path"

### 生成预测值
cd "$project_dir/backtest"
python3 prediction_master_to_pred.py --data_path="$data_path" --market_name="$market_name" --is_batch=True

### 回测
cd "$project_dir/backtest"
python3 qlib_backtest.py --market_name="$market_name" --data_path="$data_path" --is_batch=True