#!/usr/bin/env bash
set -e
set -x

###############################################################################
# MASTER 滚动增量训练脚本（多轮自动串联）
#
# 思路：
# - 每一轮 roll 生成一个新的数据目录（新的 folder_name），目录里包含：
#   - workflow_config_master_Alpha158_${market_name}.yaml（已写入本轮日期段）
#   - ${market_name}_self_dl_{train,valid,test}.pkl
# - 训练时 rolling=True，从上一轮的 Master_results 中加载 ckpt（warm-start）
# - 增量训练通常建议：更少 epoch、更小 lr（用 *override 参数实现，不改 yaml）
#
# 重要提醒：
# - 每一轮必须用新的 folder_name（避免覆盖/污染旧 ckpt）
# - 你们当前实现是 warm-start（只加载权重），不是断点续训（不会恢复 optimizer 状态）
###############################################################################

### 路径与环境
project_dir="/home/idc2/notebook/zxf/code"
data_dir="/home/idc2/notebook/zxf/data"
market_name="csi800"
qlib_path="/home/idc2/notebook/qlib_bin/cn_data_train"
conda_env="pytorch"

### 滚动计划
# roll_mode:
# - shift：train/valid/test 的 start/end 全部整体平移（保持窗口长度和间隔不变）
# - expand_train：train_start 固定，train_end 前移；valid/test 仍整体平移
roll_mode="shift"
roll_stride_days=5     # 每轮向前滚动的天数（自然日，使用 `date -d`）
roll_count=3           # 总共跑多少轮

### 初始日期段（roll=0 的日期段；后续按 roll_mode + stride 推进）
train_start_0="2015-01-01"
train_end_0="2023-04-30"
valid_start_0="2023-05-01"
valid_end_0="2024-04-30"
test_start_0="2025-05-01"
test_end_0="2025-12-31"

### warm-start 初始化（roll=0 的起点来自 A 实验）
# 你需要把这里改成你选好的 A 实验最优 seed / ckpt 目录 / step
init_seed=67
init_dir="/home/idc2/notebook/zxf/data/master_results/csi800_20260105_f1_20150101_20251231/Master_results"
init_step="28"                 # 可留空：默认取该 seed 的最大 step

### 每轮增量训练的超参（建议小步快跑）
n_epochs_override="3"           # 每轮只训练 1~5 个 epoch（视数据增量大小而定）
lr_override="0.0000012"         # 通常比原 lr 小 5~10 倍更稳
train_stop_loss_thred_override=""   # 可留空：使用 yaml 默认
strict_load="True"              # 建议保持 True：特征/结构不一致时直接报错

### init_step 选择策略
# - last：每轮使用上一轮的最大 step（默认行为；不传 init_step 即可）
# - best_valid_ic：每轮训练后从 train_metrics_results.csv 选 Valid_IC 最大的 step 作为下一轮 init_step
init_step_mode="best_valid_ic"


date_add_days() {
  # 输入: YYYY-MM-DD, 天数偏移
  # 输出: YYYY-MM-DD
  local d="$1"
  local n="$2"
  date -d "${d} ${n} days" +"%Y-%m-%d"
}

to_ymd() {
  # YYYY-MM-DD -> YYYYMMDD
  echo "${1//-/}"
}

pick_best_valid_ic_step() {
  # 从某个 Master_results/train_metrics_results.csv 中，挑指定 seed 的 Valid_IC 最大 step
  # 用法：pick_best_valid_ic_step <train_metrics_csv> <seed>
  local csv_path="$1"
  local seed="$2"
  python3 - <<PY
import pandas as pd
csv_path = "${csv_path}"
seed = int("${seed}")
df = pd.read_csv(csv_path)
df = df[df["Seed"] == seed]
if df.empty:
    raise SystemExit(f"no rows for seed={seed} in {csv_path}")
row = df.loc[df["Valid_IC"].idxmax()]
print(int(row["Step"]))
PY
}


### 激活环境
cd "$project_dir"
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate "$conda_env"

if [ ! -d "$init_dir" ]; then
  echo "[error] init_dir not found: ${init_dir}"
  exit 1
fi

prev_init_dir="$init_dir"
prev_init_step="$init_step"

# 如果选择 best_valid_ic 且未手工指定 init_step，则尝试从初始目录的 train_metrics_results.csv 自动挑选
if [ "$init_step_mode" = "best_valid_ic" ] && [ -z "$prev_init_step" ]; then
  init_metrics_csv="${prev_init_dir}/train_metrics_results.csv"
  if [ -f "$init_metrics_csv" ]; then
    prev_init_step="$(pick_best_valid_ic_step "$init_metrics_csv" "$init_seed")"
    echo "[roll] initial init_step (best_valid_ic)=${prev_init_step}"
  else
    echo "[roll][warn] init_step_mode=best_valid_ic but metrics csv not found: ${init_metrics_csv}; fallback to last step."
  fi
fi

for ((i=0; i<roll_count; i++)); do
  offset_days=$((i * roll_stride_days))

  if [ "$roll_mode" = "shift" ]; then
    train_start="$(date_add_days "$train_start_0" "$offset_days")"
  else
    train_start="$train_start_0"
  fi
  train_end="$(date_add_days "$train_end_0" "$offset_days")"
  valid_start="$(date_add_days "$valid_start_0" "$offset_days")"
  valid_end="$(date_add_days "$valid_end_0" "$offset_days")"
  test_start="$(date_add_days "$test_start_0" "$offset_days")"
  test_end="$(date_add_days "$test_end_0" "$offset_days")"

  folder_name="${market_name}_inc_r${i}_$(to_ymd "$train_start")_$(to_ymd "$test_end")"
  echo "[roll] i=${i} folder_name=${folder_name} offset_days=${offset_days}"
  echo "[roll] train=${train_start}~${train_end} valid=${valid_start}~${valid_end} test=${test_start}~${test_end}"
  echo "[roll] init_seed=${init_seed} init_dir=${prev_init_dir} init_step=${prev_init_step}"

  ### 1) 生成本轮数据（写入本轮目录的 workflow_config + dl_*.pkl）
  cd "$project_dir"
  python3 data_generator.py \
    --market_name="$market_name" \
    --qlib_path="$qlib_path" \
    --data_path="$data_dir" \
    --folder_name="$folder_name" \
    --train_start="$train_start" --train_end="$train_end" \
    --valid_start="$valid_start" --valid_end="$valid_end" \
    --test_start="$test_start" --test_end="$test_end"

  ### 2) 训练（rolling=True，从上一轮 ckpt warm-start）
  cd "$project_dir/Master"

  extra_args=""
  if [ -n "$prev_init_step" ]; then
    extra_args="${extra_args} --init_step=${prev_init_step}"
  fi
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

  python3 main.py \
    --market_name="$market_name" \
    --folder_name="$folder_name" \
    --seed_num=1 \
    --data_path="$data_dir" \
    --rolling=True \
    --init_seed="$init_seed" \
    --init_dir="$prev_init_dir" \
    $extra_args

  ### 3) 更新下一轮 warm-start 来源
  prev_init_dir="${data_dir}/master_results/${folder_name}/Master_results"
  prev_init_step=""

  if [ "$init_step_mode" = "best_valid_ic" ]; then
    metrics_csv="${data_dir}/master_results/${folder_name}/Master_results/train_metrics_results.csv"
    prev_init_step="$(pick_best_valid_ic_step "$metrics_csv" "$init_seed")"
    echo "[roll] next init_step (best_valid_ic)=${prev_init_step}"
  fi
done

echo "[done] rolling incremental training finished."


