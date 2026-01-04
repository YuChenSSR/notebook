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

# 替换 YAML 文件中的当天日期占位符
if universe == 'csi1000':
    sed -i -e s/{{today}}/$today/g -e s/{{universe}}/$universe/g workflow_config_master_Alpha158_4_last_day_predictions_1000.yaml
else:
    sed -i -e s/{{today}}/$today/g -e s/{{universe}}/$universe/g workflow_config_master_Alpha158_4_last_day_predictions.yaml

# 执行数据划分
python3 data_generator_4_last_day_predictions_1000.py 

echo "完成数据划分"

# 执行预测
python3 predict_and_send_result.py

echo "完成预测"

# 还原 YAML 文件中的当天日期占位符
if universe == 'csi1000':
    sed -i -e s/$today/{{today}}/g -e s/$universe/{{universe}}/g workflow_config_master_Alpha158_4_last_day_predictions_1000.yaml
else:
    sed -i -e s/$today/{{today}}/g -e s/$universe/{{universe}}/g workflow_config_master_Alpha158_4_last_day_predictions.yaml

echo "YAML文件还原完成"

# 删除每天整合的数据文件（每天大于2GB，占用存储空间，每次执行后删除比较稳妥）
today_num=$(echo "$today" | tr -d '-')

# 人工指定日期
# today="20250327"

rm handler_20080101_${today_num}.pkl
rm handler_20160101_${today_num}.pkl

echo "临时数据已删除"