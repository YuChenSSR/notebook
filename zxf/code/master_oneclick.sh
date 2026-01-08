set -e
set -x

### 参数
project_dir="/home/idc2/notebook/zxf/code"
data_dir="/home/idc2/notebook/zxf/data"
market_name="csi800"
seed_num=1
today=$(date +%Y%m%d)

### GPU/性能优化（可选，不填则使用默认/配置文件）
# 提示：如果你想强制指定某张卡，推荐用 CUDA_VISIBLE_DEVICES + YAML 里 GPU=0 的组合。
cuda_visible_devices=""    # 例如 "0" 或 "1" 或 "0,1"；留空则不设置
# 说明：
# - “默认可复现 + 提速”：保持 deterministic=True，AMP/TF32 默认不开（避免改变数值精度），但打开 DataLoader 并行与 pin_memory
# - 如需进一步提速（允许数值有轻微变化但仍可复现）：再手动设置 tf32=True / amp=True
amp=""                     # True / False；默认留空（不改数值精度）
amp_dtype=""               # bf16 / fp16；默认留空
tf32=""                    # True / False；默认留空（TF32 会改变数值精度）
deterministic="True"       # True / False；默认 True（更可复现）
num_workers="4"            # 0=单进程；默认 4（一般能显著减少喂数瓶颈）
pin_memory="True"          # True / False；默认 True（GPU 训练建议开启）
persistent_workers="True"  # True / False；默认 True（num_workers>0 时有效）
prefetch_factor="2"        # 默认 2（num_workers>0 时有效）

### 是否重新生成实验数据（原脚本默认不切数据，这里保持兼容）
gen_data="False"        # True / False

### 滚动切数据（增量训练）可选覆盖
# 说明：
# - 不填则使用 ${data_dir}/workflow_config_master_Alpha158_${market_name}.yaml 的日期段
# - 填了则 data_generator.py 会写入到本次实验目录下的 workflow_config，确保可复现
config_path=""   # 可选：指定某个 yaml 作为本次切数据的配置（优先级最高）
train_start=""   # 例如 2015-01-01
train_end=""     # 例如 2023-04-30
valid_start=""   # 例如 2023-05-01
valid_end=""     # 例如 2024-04-30
test_start=""    # 例如 2025-05-01
test_end=""      # 例如 2025-12-31

### 滚动实验（warm-start）参数
# A实验：rolling="False" 正常跑多seed
# B实验：rolling="True"  + 手工选最优 init_seed + 指定 init_dir 或 init_param_path
rolling="False"         # True / False
seed="67"                 # rolling=False 时可选：只跑一个 seed
init_seed=""            # rolling=True 时必填：人工选出的最优 seed
init_dir=""             # rolling=True 时可选：上次实验的 Master_results 目录
init_param_path=""      # rolling=True 时可选：直接指定 ckpt 文件路径（优先级更高）
init_step=""            # rolling=True 时可选：不填默认用该 seed 的最大 step

### 增量训练常用参数覆盖（可选；不填则使用 workflow_config 里的默认值）
# rolling=True 时一般建议：更小 lr + 更少 epoch（例如 1~5）
n_epochs_override=""                 # 例如 5
lr_override=""                       # 例如 0.0000012
train_stop_loss_thred_override=""    # 例如 0.2（保持不变也可以）
strict_load="True"                   # True / False（建议保持 True，避免特征/结构不一致隐性出错）

### 激活虚拟环境
cd "$project_dir"
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate pytorch

### 可选：绑定可见 GPU（不改变 YAML 里的 GPU 序号语义）
export CUDA_DEVICE_ORDER=PCI_BUS_ID
if [ -n "$cuda_visible_devices" ]; then
    export CUDA_VISIBLE_DEVICES="$cuda_visible_devices"
    echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi

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
    dg_args="--market_name=${market_name} --folder_name=${folder_name} --data_path=${data_dir}"
    if [ -n "$config_path" ]; then
        dg_args="${dg_args} --config_path=${config_path}"
    fi
    if [ -n "$train_start" ]; then dg_args="${dg_args} --train_start=${train_start}"; fi
    if [ -n "$train_end" ]; then dg_args="${dg_args} --train_end=${train_end}"; fi
    if [ -n "$valid_start" ]; then dg_args="${dg_args} --valid_start=${valid_start}"; fi
    if [ -n "$valid_end" ]; then dg_args="${dg_args} --valid_end=${valid_end}"; fi
    if [ -n "$test_start" ]; then dg_args="${dg_args} --test_start=${test_start}"; fi
    if [ -n "$test_end" ]; then dg_args="${dg_args} --test_end=${test_end}"; fi
    python3 data_generator.py ${dg_args}
    # data_generator.py 现在会将实际使用的 workflow_config 写入实验目录；
    # 这里仅在目录内不存在时再拷贝默认配置，避免覆盖滚动日期/自定义配置。
    if [ ! -f "${data_dir}/master_results/${folder_name}/workflow_config_master_Alpha158_${market_name}.yaml" ]; then
        cp ${data_dir}/workflow_config_master_Alpha158_${market_name}.yaml ${data_dir}/master_results/${folder_name}/
    fi
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
    # rolling 场景下可选覆盖训练超参
    if [ -n "$n_epochs_override" ]; then
        extra_args="${extra_args} --n_epochs_override=${n_epochs_override}"
    fi
    if [ -n "$lr_override" ]; then
        extra_args="${extra_args} --lr_override=${lr_override}"
    fi
    if [ -n "$train_stop_loss_thred_override" ]; then
        extra_args="${extra_args} --train_stop_loss_thred_override=${train_stop_loss_thred_override}"
    fi
    if [ -n "$strict_load" ]; then
        extra_args="${extra_args} --strict_load=${strict_load}"
    fi
else
    if [ -n "$seed" ]; then
        extra_args="${extra_args} --seed=${seed}"
    fi
fi

# 追加性能参数（仅在变量非空时覆盖）
if [ -n "$amp" ]; then extra_args="${extra_args} --amp=${amp}"; fi
if [ -n "$amp_dtype" ]; then extra_args="${extra_args} --amp_dtype=${amp_dtype}"; fi
if [ -n "$tf32" ]; then extra_args="${extra_args} --tf32=${tf32}"; fi
if [ -n "$deterministic" ]; then extra_args="${extra_args} --deterministic=${deterministic}"; fi
if [ -n "$num_workers" ]; then extra_args="${extra_args} --num_workers=${num_workers}"; fi
if [ -n "$pin_memory" ]; then extra_args="${extra_args} --pin_memory=${pin_memory}"; fi
if [ -n "$persistent_workers" ]; then extra_args="${extra_args} --persistent_workers=${persistent_workers}"; fi
if [ -n "$prefetch_factor" ]; then extra_args="${extra_args} --prefetch_factor=${prefetch_factor}"; fi

# -u/UNBUFFERED：确保 Python 输出实时刷出（避免“看不到任何结果但其实在跑”）
PYTHONUNBUFFERED=1 python3 -u main.py --market_name="$market_name" --folder_name="$folder_name" --seed_num="$seed_num" --data_path="$data_dir" $extra_args

# ### 生成预测值
# cd "$project_dir/Backtest"
# python3 processing_prediction_seed_step.py --data_path="$data_dir" --market_name="$market_name" --folder_name="$folder_name"

# ### 回测
# cd "$project_dir/Backtest"
# python3 backtest_seed_step.py --market_name="$market_name" --folder_name="$folder_name"