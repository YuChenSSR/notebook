set -e
set -x

### 参数
project_dir="/home/idc2/notebook/zxf/code"
data_dir="/home/idc2/notebook/zxf/data"
market_name="csi800"
seed_num=2
today=$(date +%Y%m%d)

### 是否重新生成实验数据（原脚本默认不切数据，这里保持兼容）
gen_data="False"        # True / False

### 滚动实验（warm-start）参数
# A实验：rolling="False" 正常跑多seed
# B实验：rolling="True"  + 手工选最优 init_seed + 指定 init_dir 或 init_param_path
rolling="False"         # True / False
seed=""                 # rolling=False 时可选：只跑一个 seed
init_seed=""            # rolling=True 时必填：人工选出的最优 seed
init_dir=""             # rolling=True 时可选：上次实验的 Master_results 目录
init_param_path=""      # rolling=True 时可选：直接指定 ckpt 文件路径（优先级更高）
init_step=""            # rolling=True 时可选：不填默认用该 seed 的最大 step

### 激活虚拟环境
cd "$project_dir"
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate pytorch

### 读取workflow_config参数
cd "$project_dir"
# read -r start_date end_date <<< $(python - <<EOF
# import yaml
# from datetime import datetime

# with open("${data_dir}/workflow_config_master_Alpha158_${market_name}.yaml", 'r') as f:
#     config = yaml.safe_load(f)

# start_date = config["task"]["dataset"]["kwargs"]["segments"]["train"][0].strftime("%Y%m%d")
# end_date = config["task"]["dataset"]["kwargs"]["segments"]["test"][1].strftime("%Y%m%d")
# print(f"{start_date} {end_date}")
# EOF
# )
# if [ -z "$start_date" ] || [ -z "$end_date" ]; then
#     echo "Error: Python script failed to extract dates."
#     exit 1
# fi


### 生成实验主目录
# folder_name="${market_name}_${today}_${start_date}_${end_date}"
folder_name="csi800_20260105_f1_20150101_20251231"
echo "Folder_name:$folder_name"

### 数据切割
if [ "$gen_data" = "True" ]; then
    cd "$project_dir"
    python3 data_generator.py --market_name="$market_name" --folder_name="$folder_name"
    cp ${data_dir}/workflow_config_master_Alpha158_${market_name}.yaml ${data_dir}/master_results/${folder_name}/
else
    echo "Skip data_generator (gen_data=False)."
fi

### 主实验
cd "$project_dir/Master"

extra_args=""
if [ "$rolling" = "True" ]; then
    if [ -z "$init_seed" ]; then
        echo "Error: rolling=True requires init_seed"
        exit 1
    fi
    if [ -z "$init_param_path" ] && [ -z "$init_dir" ]; then
        echo "Error: rolling=True requires init_param_path or init_dir"
        exit 1
    fi

    extra_args="${extra_args} --rolling=True --init_seed=${init_seed}"
    if [ -n "$init_param_path" ]; then
        extra_args="${extra_args} --init_param_path=${init_param_path}"
    else
        extra_args="${extra_args} --init_dir=${init_dir}"
    fi
    if [ -n "$init_step" ]; then
        extra_args="${extra_args} --init_step=${init_step}"
    fi
else
    if [ -n "$seed" ]; then
        extra_args="${extra_args} --seed=${seed}"
    fi
fi

python3 main.py --market_name="$market_name" --folder_name="$folder_name" --seed_num="$seed_num" --data_path="$data_dir" $extra_args

### 生成预测值
cd "$project_dir/Backtest"
python3 processing_prediction_seed_step.py --data_path="$data_dir" --market_name="$market_name" --folder_name="$folder_name"

### 回测
cd "$project_dir/Backtest"
python3 backtest_seed_step.py --market_name="$market_name" --folder_name="$folder_name"