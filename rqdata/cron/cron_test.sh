#!/bin/bash


### 环境配置 #############################################################################
set -e
set -x

# 切换虚拟环境
# export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/opt/anaconda3/bin

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate pytorch



### 
cd "/home/idc2/notebook/rqdata/test"


MAX_JOBS=5

TASKS_LIST=(
    "python3 a.py"
    "python3 b.py" 
    "python3 c.py"
    "python3 d.py"
    "python3 e.py"
    "python3 f.py"
    "python3 g.py"
    "python3 h.py"
    "python3 i.py"
)

echo "开始执行 ${#TASKS_LIST[@]} 个任务，最大并发数: $MAX_JOBS"

# 启动初始的MAX_JOBS个任务
for ((i=0; i < MAX_JOBS && i < ${#TASKS_LIST[@]}; i++)); do
    echo "启动任务: ${TASKS_LIST[i]}"
    eval "${TASKS_LIST[i]}" &
done

# 记录下一个要启动的任务索引
next_index=$MAX_JOBS

# 循环处理剩余任务
while [ $next_index -lt ${#TASKS_LIST[@]} ]; do
    # 等待任意一个任务完成
    wait -n
    # 启动下一个任务
    echo "启动任务: ${TASKS_LIST[next_index]}"
    eval "${TASKS_LIST[next_index]}" &
    ((next_index++))
done

# 等待所有剩余任务完成
wait

echo "所有任务执行完成"


# 将时间转换为秒数（从当天00:00:00开始计算的秒数）
current_seconds=$((10#$(date -d "$(date +%H:%M:%S)" +%s) - 10#$(date -d "00:00:00" +%s)))
target_seconds=$((10#$(date -d "20:15:00" +%s) - 10#$(date -d "00:00:00" +%s)))

# 判断当前时间是否早于20:05:00
if [ $current_seconds -lt $target_seconds ]; then
    # 计算需要sleep的秒数
    sleep_seconds=$((target_seconds - current_seconds))
    echo "当前时间早于20:05:00，需要等待 ${sleep_seconds} 秒"
    echo "等待至20:05:00..."
    
    # 使用sleep等待
    sleep $sleep_seconds
    
    echo "等待完成，当前时间: $(date '+%H:%M:%S')"
else
    echo "当前时间已晚于或等于20:05:00，直接执行任务"
fi

echo "hahahah"
