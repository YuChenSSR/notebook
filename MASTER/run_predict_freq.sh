#!/bin/bash
set -e

# 获取日期参数
today=$1
echo $today

# 获取股票池参数
universe=$2
echo $universe

# 人工指定日期
# today="2025-04-02"
# echo ${today}

# 根据 universe 参数选择对应的 YAML 文件
if [ "$universe" = "csi800" ]; then
    config_file="workflow_config_master_Alpha158_4_last_day_predictions_csi800.yaml"
elif [ "$universe" = "csi1000" ]; then
    config_file="workflow_config_master_Alpha158_4_last_day_predictions_csi1000.yaml"
elif [ "$universe" = "csi300" ]; then
    config_file="workflow_config_master_Alpha158_4_last_day_predictions_csi300.yaml"
else
    # 默认配置或其他股票池配置
    config_file="workflow_config_master_Alpha158_4_last_day_predictions.yaml"
fi

echo "使用配置文件: $config_file"

# 替换 YAML 文件中的当天日期占位符
sed -i -e "s/{{today}}/$today/g" -e "s/{{universe}}/$universe/g" "$config_file"

# 执行数据划分
if [ "$universe" = "csi800" ]; then
    python3 data_generator_4_last_day_predictions.py --universe csi800
elif [ "$universe" = "csi1000" ]; then
    python3 data_generator_4_last_day_predictions.py --universe csi1000
elif [ "$universe" = "csi300" ]; then
    python3 data_generator_4_last_day_predictions.py --universe csi300
else
    # 默认配置或其他股票池配置
    python3 data_generator_4_last_day_predictions.py --universe default
fi

echo "完成数据划分"

# 执行预测
if [ "$universe" = "csi800" ]; then
    python3 predict_and_send_result_freq.py csi800
elif [ "$universe" = "csi1000" ]; then
    python3 predict_and_send_result_freq.py csi1000
elif [ "$universe" = "csi300" ]; then
    python3 predict_and_send_result_freq.py csi300
else
    # 默认配置或其他股票池配置
    python3 predict_and_send_result_freq.py other
fi

echo "完成预测"

# 还原 YAML 文件中的当天日期占位符
sed -i -e "s/$today/{{today}}/g" -e "s/$universe/{{universe}}/g" "$config_file"

echo "YAML文件还原完成"


# 删除每天整合的数据文件（每天大于2GB，占用存储空间，每次执行后删除比较稳妥）
today_num=$(echo "$today" | tr -d '-')

# 人工指定日期
# today="20250327"

if [ "$universe" = "csi800" ]; then
    rm handler_20120101_${today_num}.pkl
elif [ "$universe" = "csi1000" ]; then
    rm handler_20160101_${today_num}.pkl
else
    rm handler_20080101_${today_num}.pkl
fi

# rm handler_20080101_${today_num}.pkl
# rm handler_20120101_${today_num}.pkl
# rm handler_20160101_${today_num}.pkl

echo "临时数据已删除"