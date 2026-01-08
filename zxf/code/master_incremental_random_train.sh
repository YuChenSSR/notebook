#!/usr/bin/env bash
set -e
set -x

###############################################################################
# 增量训练方案：随机新增 Δ 天（交易日） + 回放过去 W 天（交易日） 续训
#
# 每轮 k：
# - 随机抽 Δ_k ∈ [1, max_add_td]
# - 将 train_end 向后推进 Δ_k 个交易日
# - 训练窗口：最近 W 个交易日（含最新的 Δ_k 天）=> train_start = train_end - (W-1)
# - valid/test：各取 V/T 个交易日紧随其后（无泄漏）
# - rolling=True warm-start：从上一轮 ckpt 加载，训练少量 epoch（建议 1~5）
#
# 说明：
# - 这是“随机到达数据 + 有限回放”的在线/增量模拟；不会再用 2015~2025 全量窗口。
# - 你需要先有一个初始 ckpt（通常来自 A 实验的 best seed/step）。
###############################################################################

### 路径与环境
project_dir="/home/idc2/notebook/zxf/code"
data_dir="/home/idc2/notebook/zxf/data"
market_name="csi800"
qlib_path="/home/idc2/notebook/qlib_bin/cn_data_train"
conda_env="pytorch"

### warm-start 初始点（来自 A 实验）
init_seed=67
init_dir="/home/idc2/notebook/zxf/data/master_results/csi800_20260105_f1_20150101_20251231/Master_results"
init_step="28"   # 可留空：默认取 init_dir 中该 seed 最大 step

### 基准配置（必须与 warm-start ckpt 的实验一致）
config_path=""   # 留空则从 init_dir 上级目录推断

### 增量窗口参数（交易日）
replay_window_td=60     # 回放窗口 W：建议 60~252；越大越稳但越慢/越吃内存
valid_window_td=20      # V
test_window_td=20       # T
max_add_td=5            # Δ_k 的上界：每轮随机新增 1~max_add_td 个交易日
roll_count=10           # 总轮数
random_seed=20260109    # 控制随机性，可复现

### 性能/加速（不改变训练结果）
# 经验：每轮总耗时里 data_generator 占大头（CPU+I/O），训练只占小头（GPU）
# 因此可以先把所有轮次的数据并行生成好，再按顺序做 warm-start 训练，显著减少总墙钟时间。
pre_generate_data="true"    # true: 先并行生成所有轮次数据，再按顺序训练（推荐）
gen_parallel=4              # 并行 data_generator 进程数（建议 2~8；内存/IO不够就调小）
# skip_existing_data="true"   # true: 若本轮 dl_*.pkl 已存在则跳过生成（便于断点续跑）
skip_existing_data="false"
# 传给 data_generator.py（python-fire 更稳妥地识别 True/False）
if [ "${skip_existing_data}" = "true" ]; then
  dg_skip_if_exists="True"
else
  dg_skip_if_exists="False"
fi

### 每轮训练超参（建议小步快跑）
n_epochs_override="3"   # 每轮 1~5
lr_override="0.0000012"
strict_load="True"
init_step_mode="best_valid_ic"  # last / best_valid_ic

### 初始 train_end（增量从哪个时间点开始）
# 建议设为“你想开始在线更新的那天”（需要在 qlib 交易日历内）
train_end_0="2025-04-30"


pick_best_valid_ic_step() {
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

if [ -z "$config_path" ]; then
  inferred_cfg="$(dirname "$init_dir")/workflow_config_master_Alpha158_${market_name}.yaml"
  if [ -f "$inferred_cfg" ]; then
    config_path="$inferred_cfg"
    echo "[inc] inferred config_path=${config_path}"
  fi
fi
if [ -z "$config_path" ] || [ ! -f "$config_path" ]; then
  echo "[error] config_path not found. 请设置 config_path 指向 A 实验的 workflow_config yaml。"
  echo "        expected: $(dirname "$init_dir")/workflow_config_master_Alpha158_${market_name}.yaml"
  exit 1
fi

prev_init_dir="$init_dir"
prev_init_step="$init_step"

### 预先生成“随机增量日程”（按交易日历推进）
schedule_file="/tmp/master_inc_schedule_${market_name}_$$.tsv"
python3 - <<PY > "$schedule_file"
import random, bisect
import pandas as pd
import qlib
from qlib.constant import REG_CN
from qlib.data import D

qlib.init(provider_uri="${qlib_path}", region=REG_CN)

max_add_td = int("${max_add_td}")
roll_count = int("${roll_count}")
replay_window_td = int("${replay_window_td}")
valid_window_td = int("${valid_window_td}")
test_window_td = int("${test_window_td}")
seed = int("${random_seed}")

rng = random.Random(seed)

def _ts(x): return pd.Timestamp(str(x))

# 获取足够长的交易日历（尽量覆盖未来 V/T 窗口）
cal = list(D.calendar(start_time="2000-01-01", end_time="2030-12-31", freq="day"))

def align_end(d):
    """对齐到 <= d 的最近交易日"""
    t = _ts(d)
    i = bisect.bisect_right(cal, t) - 1
    if i < 0:
        raise ValueError(f"date too early: {d}")
    return i

def add_td(idx, n):
    j = idx + n
    if j < 0 or j >= len(cal):
        raise IndexError("calendar out of range")
    return j

train_end_idx = align_end("${train_end_0}")

print("roll\tdelta_td\ttrain_start\ttrain_end\tvalid_start\tvalid_end\ttest_start\ttest_end")
for r in range(roll_count):
    delta = rng.randint(1, max_add_td)
    train_end_idx = add_td(train_end_idx, delta)
    train_start_idx = add_td(train_end_idx, -(replay_window_td - 1))

    valid_start_idx = add_td(train_end_idx, 1)
    valid_end_idx = add_td(valid_start_idx, valid_window_td - 1)
    test_start_idx = add_td(valid_end_idx, 1)
    test_end_idx = add_td(test_start_idx, test_window_td - 1)

    def fmt(i): return cal[i].strftime("%Y-%m-%d")
    print(f"{r}\t{delta}\t{fmt(train_start_idx)}\t{fmt(train_end_idx)}\t{fmt(valid_start_idx)}\t{fmt(valid_end_idx)}\t{fmt(test_start_idx)}\t{fmt(test_end_idx)}")
PY

echo "[inc] schedule written: ${schedule_file}"
head -n 5 "$schedule_file" || true

###############################################################################
# 0) 可选：先并行生成所有轮次的数据（数据生成彼此独立，可并行；训练仍需顺序 warm-start）
###############################################################################
if [ "${pre_generate_data}" = "true" ]; then
  echo "[inc] pre-generating datasets with parallel=${gen_parallel} (skip_existing_data=${skip_existing_data}) ..."

  # 导出 CONDA_PREFIX 让子进程能用 conda 环境里的 python
  export project_dir data_dir market_name qlib_path config_path replay_window_td skip_existing_data dg_skip_if_exists CONDA_PREFIX
  # schedule_file 为 tsv：roll delta train_start train_end valid_start valid_end test_start test_end
  tail -n +2 "$schedule_file" | tr '\t' ' ' | xargs -P "${gen_parallel}" -n 8 bash -c '
    set -e
    roll="$1"; delta_td="$2"; train_start="$3"; train_end="$4"; valid_start="$5"; valid_end="$6"; test_start="$7"; test_end="$8"
    folder_name="${market_name}_inc_rand_r${roll}_te${train_end//-/}_d${delta_td}_W${replay_window_td}"
    save_path="${data_dir}/master_results/${folder_name}"

    if [ "${skip_existing_data}" = "true" ] && \
       [ -f "${save_path}/${market_name}_self_dl_train.pkl" ] && \
       [ -f "${save_path}/${market_name}_self_dl_valid.pkl" ] && \
       [ -f "${save_path}/${market_name}_self_dl_test.pkl" ]; then
      echo "[inc][datagen-skip] ${folder_name}"
      exit 0
    fi

    echo "[inc][datagen] roll=${roll} folder_name=${folder_name}"
    cd "${project_dir}"
    # 并行时建议限制每个进程的 BLAS/OMP 线程数，避免 CPU 线程过度竞争导致更慢
    # 使用 CONDA_PREFIX 里的 python，确保子进程也用 conda 环境
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 "${CONDA_PREFIX}/bin/python3" data_generator.py \
      --market_name="${market_name}" \
      --qlib_path="${qlib_path}" \
      --data_path="${data_dir}" \
      --folder_name="${folder_name}" \
      --config_path="${config_path}" \
      --handler_dump_all=False \
      --train_start="${train_start}" --train_end="${train_end}" \
      --valid_start="${valid_start}" --valid_end="${valid_end}" \
      --test_start="${test_start}" --test_end="${test_end}" \
      --skip_if_exists="${dg_skip_if_exists}"
  ' _

  echo "[inc] pre-generate datasets done."
fi

###############################################################################
# 1) 顺序 warm-start 训练（必须按 roll 顺序，因为下一轮要接上一轮 ckpt）
###############################################################################
tail -n +2 "$schedule_file" | while IFS=$'\t' read -r roll delta_td train_start train_end valid_start valid_end test_start test_end; do
  folder_name="${market_name}_inc_rand_r${roll}_te${train_end//-/}_d${delta_td}_W${replay_window_td}"
  echo "[inc] roll=${roll} delta_td=${delta_td} folder_name=${folder_name}"
  echo "[inc] train=${train_start}~${train_end} valid=${valid_start}~${valid_end} test=${test_start}~${test_end}"
  echo "[inc] init_seed=${init_seed} init_dir=${prev_init_dir} init_step=${prev_init_step}"

  ### 1) 确保本轮小窗口数据已就绪（如果上面未并行预生成或中途失败，这里兜底生成）
  save_path="${data_dir}/master_results/${folder_name}"
  need_gen="false"
  if [ "${pre_generate_data}" = "true" ]; then
    # 预生成模式下这里仅做兜底：缺文件才生成
    if [ ! -f "${save_path}/${market_name}_self_dl_train.pkl" ] || \
       [ ! -f "${save_path}/${market_name}_self_dl_valid.pkl" ] || \
       [ ! -f "${save_path}/${market_name}_self_dl_test.pkl" ]; then
      need_gen="true"
    fi
  else
    # 非预生成模式：由 skip_existing_data 决定是否复用历史数据
    if [ "${skip_existing_data}" = "true" ] && \
       [ -f "${save_path}/${market_name}_self_dl_train.pkl" ] && \
       [ -f "${save_path}/${market_name}_self_dl_valid.pkl" ] && \
       [ -f "${save_path}/${market_name}_self_dl_test.pkl" ]; then
      need_gen="false"
    else
      need_gen="true"
    fi
  fi

  if [ "${need_gen}" = "true" ]; then
    echo "[inc] generating dataset: ${folder_name} (skip_if_exists=${dg_skip_if_exists})"
    cd "$project_dir"
    python3 data_generator.py \
      --market_name="$market_name" \
      --qlib_path="$qlib_path" \
      --data_path="$data_dir" \
      --folder_name="$folder_name" \
      --config_path="$config_path" \
      --handler_dump_all=False \
      --train_start="$train_start" --train_end="$train_end" \
      --valid_start="$valid_start" --valid_end="$valid_end" \
      --test_start="$test_start" --test_end="$test_end" \
      --skip_if_exists="${dg_skip_if_exists}"
  else
    echo "[inc] dataset ready: ${folder_name}"
  fi

  ### 2) 增量续训（rolling warm-start）
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

  ### 3) 下一轮 warm-start 来源更新（使用本轮输出）
  prev_init_dir="${data_dir}/master_results/${folder_name}/Master_results"
  prev_init_step=""
  if [ "$init_step_mode" = "best_valid_ic" ]; then
    metrics_csv="${prev_init_dir}/train_metrics_results.csv"
    prev_init_step="$(pick_best_valid_ic_step "$metrics_csv" "$init_seed")"
    echo "[inc] next init_step (best_valid_ic)=${prev_init_step}"
  fi
done

echo "[done] incremental random training finished."


