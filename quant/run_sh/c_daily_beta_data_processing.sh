#!/bin/bash
set -e
set -x

### 参数
code_dir="/home/a/notebook/zxf/code"
data_dir="/home/a/notebook/zxf/data"
folder_name="model_232_128"
seed=87
step=48
d_model=128

data_path="$data_dir/Daily_data/Beta_data/$folder_name"

### 解压tar包 ###
cd "/home/a/notebook"
tar -xzf cn_data_train.tar.gz

########## 生成csi800 ########## ########## ##########
market_name="csi800"
# # 数据切割
cd "$code_dir/data_generator"
python3 data_generator.py --market_name="$market_name" --data_path="$data_path" --folder_name="$market_name"
# 生成beta数据
cd "$code_dir/beta_data_processing"
python3 beta_data_processing.py --market_name="$market_name" --data_path="$data_path" --seed="$seed" --step="$step" --d_model="$d_model" 


########## 生成csi800 ########## ########## ##########
market_name="csi800b"
# # 数据切割
cd "$code_dir/data_generator"
python3 data_generator.py --market_name="$market_name" --data_path="$data_path" --folder_name="$market_name"
# 生成beta数据
cd "$code_dir/beta_data_processing"
python3 beta_data_processing.py --market_name="$market_name" --data_path="$data_path" --seed="$seed" --step="$step" --d_model="$d_model" 


########## 生成csi800c ########## ########## ##########
market_name="csi800c"
# 加工csi800c.txt
cd "$code_dir/beta_data_processing"
python3 csi800c_processing.py
# 数据切割
cd "$code_dir/data_generator"
python3 data_generator.py --market_name="$market_name" --data_path="$data_path" --folder_name="$market_name"
# 生成beta数据
cd "$code_dir/beta_data_processing"
python3 beta_data_processing.py --market_name="$market_name" --data_path="$data_path" --seed="$seed" --step="$step" --d_model="$d_model" 
