#!/bin/bash
set -e

# 进入MASTER目录
cd ~/notebook/MASTER

# 检查是否传递了参数
if [ -z "$1" ]; then
    # 如果没有传递参数，使用当前日期
    today=$(date +%Y-%m-%d)
else
    # 如果传递了参数，使用该参数
    today="$1"
fi
echo $today

# 获取当前是星期几（1-7，1=周一，7=周日）
current_day=$(date +%u)
echo current day is $current_day

# 如果是周六（6）或周日（7），则退出脚本
if [ "$current_day" -eq 6 ] || [ "$current_day" -eq 7 ]; then
    echo "Today is weekend. Exiting..."
    exit 0
fi

# 若预测结果已存在，终止执行；若不存在，说明上次执行失败，再次执行
if [ -f ~/notebook/MASTER/master_predictions_csi800_5_single_day_${today}.csv ]; then
    echo "已完成预测，无需再次执行"
    exit 0
else
    echo "未找到预测结果，再次执行"
fi

# sh download_stock_data.sh $today
# sh run_predict.sh $today csi300
sh run_predict.sh $today csi800
# sh run_predict.sh $today csi1000