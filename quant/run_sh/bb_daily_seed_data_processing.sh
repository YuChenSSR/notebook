#!/bin/bash
set -e
set -x

### 参数
code_dir="/home/a/notebook/zxf/code"
data_dir="/home/a/notebook/zxf/data"
market_name="csi800"


### 种子seed7 seed:27-31; d_model:256
folder_name="seed7"
data_path="$data_dir/Daily_data/Good_seed/$folder_name"

# 数据切割
cd "$code_dir/data_generator"
python3 data_generator.py --market_name="$market_name" --data_path="$data_path" --folder_name="$market_name"
# 特征因子筛选
cd "$code_dir/evaluate_features"
python3 processing_dl_data.py --market_name="$market_name" --data_dir="$data_path"
# 生成预测值
cd "$code_dir/processing_prediction"
python3 processing_prediction_daily.py --data_path="$data_path" --market_name="$market_name" --folder_name="$folder_name"
# 复制文件的目标文件夹
cd "$data_path"
cp -p master_predictions_backday_8_csi800_27_31.csv /home/a/notebook/MASTER/master_predictions_csi800_7.csv


### 种子seed7 seed:91-35; d_model:512
# 复制实验文件
folder_name_2="seed8"
data_path_2="$data_dir/Daily_data/Good_seed/$folder_name_2"
cp -p "${market_name}_self_dl_test.pkl" "$data_path_2"
# 生成预测值
cd "$code_dir/processing_prediction"
python3 processing_prediction_daily.py --data_path="$data_path_2" --market_name="$market_name" --folder_name="$folder_name_2"
# 复制文件的目标文件夹
cd "$data_path_2"
cp -p master_predictions_backday_8_csi800_91_35.csv /home/a/notebook/MASTER/master_predictions_csi800_8.csv



