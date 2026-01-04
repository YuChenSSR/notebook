 #!/bin/bash

set -e
set -x

# 切换虚拟环境
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/opt/anaconda3/bin

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate pytorch

# rqdatac license
LICENSE_FILE="/home/idc2/notebook/rqdata/.rqdatac/license.txt"
# 检查许可证文件是否存在
if [ ! -f "$LICENSE_FILE" ]; then
    echo "错误：许可证文件 $LICENSE_FILE 未找到"
    exit 1
fi

IFS= read -r LICENSE_CONTENT < "$LICENSE_FILE"
if [ -z "$LICENSE_CONTENT" ]; then
    echo "错误：许可证文件内容为空"
    exit 1
fi
export RQSDK_LICENSE="tcp://license:${LICENSE_CONTENT}@rqdatad-pro.ricequant.com:16011"
export RQDATAC_CONF="tcp://license:${LICENSE_CONTENT}@rqdatad-pro.ricequant.com:16011"

### 数据下载 #############################################################################
PROJ_PATH="$HOME/notebook/rqdata"
DATA_DOWNLOAD_CODE_PATH="$PROJ_PATH/data_download"
cd "$DATA_DOWNLOAD_CODE_PATH"

echo "=== Step1: 获取并发用参数 [$(date +"%Y-%m-%d %H:%M:%S")] ==="
eval $(python3 rq_99_read_stock_list.py | grep "part[12]_")

echo "=== Step2: turn、capital_flow [$(date +"%Y-%m-%d %H:%M:%S")] ==="
python3 rq_21_stock_daily_turn.py --market=["XSHE"] &
python3 rq_21_stock_daily_turn.py --market=["XSHG","BJSE"] &

python3 rq_32_stock_daily_capital_flow.py --market=["XSHE"] &
python3 rq_32_stock_daily_capital_flow.py --market=["XSHG","BJSE"] &
wait

echo "=== Step3: factor [$(date +"%Y-%m-%d %H:%M:%S")] ==="
python3 rq_31_stock_daily_factor.py --market=["XSHE"] --index_start="$part1_paragraph1_start" --index_end="$part1_paragraph1_end" &
python3 rq_31_stock_daily_factor.py --market=["XSHE"] --index_start="$part1_paragraph2_start" --index_end="$part1_paragraph2_end" &

python3 rq_31_stock_daily_factor.py --market=["XSHG","BJSE"] --index_start="$part2_paragraph1_start" --index_end="$part2_paragraph1_end" &
python3 rq_31_stock_daily_factor.py --market=["XSHG","BJSE"] --index_start="$part2_paragraph2_start" --index_end="$part2_paragraph2_end" &
wait

echo "=== Data Download Batch2 Completed [$(date +"%Y-%m-%d %H:%M:%S")] ==="

### 数据转化QLIB #############################################################################
DUMP_CODE_PATH="$PROJ_PATH/dump_qlib"
CSV_SOURCE="$PROJ_PATH/Data/qlib_data"

QLIB_TRAIN_DATA_PATH="$PROJ_PATH/Data/qlib_data/cn_data_train"
mkdir -p "$QLIB_TRAIN_DATA_PATH"

QLIB_BACKTEST_DATA_PATH="$PROJ_PATH/Data/qlib_data/cn_data_backtest"
mkdir -p "$QLIB_BACKTEST_DATA_PATH"

cd "$DUMP_CODE_PATH"
echo "=== Step:1 合并数据 [$(date +"%Y-%m-%d %H:%M:%S")] ==="
python3 dump_s_qlib_source.py
python3 dump_i_qlib_source.py

echo "=== Step:2 数据标准化 [$(date +"%Y-%m-%d %H:%M:%S")] ==="
python3 normalize.py 

echo "=== Step:3 转换Qlib [$(date +"%Y-%m-%d %H:%M:%S")] ==="
QLIB_SCRIPTS_DIR="$HOME/notebook/qlib/scripts/"
export PYTHONPATH=$PYTHONPATH:$QLIB_SCRIPTS_DIR

# train
python3 "$QLIB_SCRIPTS_DIR/dump_bin.py" dump_all \
    --csv_path "$CSV_SOURCE/qlib_train_source" \
    --qlib_dir "$QLIB_TRAIN_DATA_PATH" \
    --date_field_name=tradedate \
    --exclude_fields=tradedate,symbol
    
# backtest
python3 "$QLIB_SCRIPTS_DIR/dump_bin.py" dump_all \
    --csv_path "$CSV_SOURCE/qlib_backtest_source" \
    --qlib_dir "$QLIB_BACKTEST_DATA_PATH" \
    --date_field_name=tradedate \
    --exclude_fields=tradedate,symbol

echo "=== Step4: 生成csi [$(date +"%Y-%m-%d %H:%M:%S")] ==="
cd "$DUMP_CODE_PATH"
python3 dump_csi.py --out_qlib_path "$QLIB_TRAIN_DATA_PATH"
python3 dump_csi_b.py --out_qlib_path "$QLIB_TRAIN_DATA_PATH"

cp $QLIB_TRAIN_DATA_PATH/instruments/csi* $QLIB_BACKTEST_DATA_PATH/instruments/

echo "=== Step5: 生成日历 [$(date +"%Y-%m-%d %H:%M:%S")] ==="
python3 dump_calendar.py --out_qlib_path "$QLIB_TRAIN_DATA_PATH"
cp $QLIB_TRAIN_DATA_PATH/calendars/day_future* $QLIB_BACKTEST_DATA_PATH/calendars/


echo "=== Step6: 复制文件 [$(date +"%Y-%m-%d %H:%M:%S")] ==="
cp -rf "$QLIB_TRAIN_DATA_PATH" /home/idc2/notebook/qlib_bin
cp -rf "$QLIB_BACKTEST_DATA_PATH" /home/idc2/notebook/qlib_bin


echo "=== Step7: 打包 [$(date +"%Y-%m-%d %H:%M:%S")] ==="
cd "$PROJ_PATH/Data/qlib_data"
tar -czf ./cn_data_train.tar.gz cn_data_train
tar -czf ./cn_data_backtest.tar.gz cn_data_backtest

echo "=== Data Dump Completed [$(date +"%Y-%m-%d %H:%M:%S")] ==="