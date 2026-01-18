#!/bin/bash
set -e
set -x

### 参数
code_dir="/home/idc2/notebook/quant"
data_dir="/home/idc2/notebook/quant/data"
market_name="csi800"
folder_name="csi800_128_20150101_20260116"

# 数据切割
# handler、dataset、workflow 需提前放在data_path目录中
# data_path="$data_dir/experimental_results/"
data_path="$data_dir/feature_engineering/$market_name/$folder_name"


cd "$code_dir/feature_processing"
python3 data_generator.py --market_name="$market_name" --data_path="$data_path"