 #!/bin/bash
 
### 环境配置 #############################################################################################################
set -e
set -x

# 切换虚拟环境
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/opt/anaconda3/bin

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate pytorch

# rqdatac license
LICENSE_FILE="/home/idc2/notebook/rqdata/.rqdatac/license.txt"
# LICENSE_FILE="/home/idc2/notebook/rqdata/cron/license_lin.txt"

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

# 并发数
MAX_JOBS=4

### 目录
PROJ_PATH="$HOME/notebook/rqdata"
DATA_DOWNLOAD_CODE_PATH="$PROJ_PATH/data_download"


### 并发用分拆参数
cd "$DATA_DOWNLOAD_CODE_PATH"
echo "=== Step1: 获取并发用分拆参数 [$(date +"%Y-%m-%d %H:%M:%S")] ==="
eval $(python3 rq_99_read_stock_list.py | grep "part[12]_")
# echo "part1_paragraph1_start: $part1_paragraph1_start"
# echo "part1_paragraph1_end: $part1_paragraph1_end"
# echo "part1_paragraph2_start: $part1_paragraph2_start"
# echo "part1_paragraph2_end: $part1_paragraph2_end"
# echo "part2_paragraph1_start: $part2_paragraph1_start"
# echo "part2_paragraph1_end: $part2_paragraph1_end"
# echo "part2_paragraph2_start: $part2_paragraph2_start"
# echo "part2_paragraph2_end: $part2_paragraph2_end"


### 数据下载batch【17:00】 #############################################################################################################

# 需优先下载
echo "=== Step2: calendar, stock列表, index列表, industry列表 [$(date +"%Y-%m-%d %H:%M:%S")] ==="
# python3 rq_0_trade_date.py &
python3 rq_1_stock_list.py &
python3 rq_2_index_list.py &
python3 rq_3_industry_list.py &
wait

# 并发处理[17点批次]
BATCH1_TASKS_LIST=(
    'python3 rq_4_index_components.py'
    'python3 rq_12_index_daily_klines.py' 
    'python3 rq_5_industry_components_step1.py'
    'python3 rq_15_stock_1m_klines.py --market=["XSHE"] --index_start="'"$part1_paragraph1_start"'" --index_end="'"$part1_paragraph1_end"'"'
    'python3 rq_15_stock_1m_klines.py --market=["XSHE"] --index_start="'"$part1_paragraph2_start"'" --index_end="'"$part1_paragraph2_end"'"'
    'python3 rq_15_stock_1m_klines.py --market=["XSHG","BJSE"] --index_start="'"$part2_paragraph1_start"'" --index_end="'"$part2_paragraph1_end"'"'
    'python3 rq_15_stock_1m_klines.py --market=["XSHG","BJSE"] --index_start="'"$part2_paragraph2_start"'" --index_end="'"$part2_paragraph2_end"'"' 
    'python3 rq_9_stock_adj_factor.py --market=["XSHE"] --index_start="'"$part1_paragraph1_start"'" --index_end="'"$part1_paragraph1_end"'"'
    'python3 rq_9_stock_adj_factor.py --market=["XSHE"] --index_start="'"$part1_paragraph2_start"'" --index_end="'"$part1_paragraph2_end"'"'
    'python3 rq_9_stock_adj_factor.py --market=["XSHG","BJSE"] --index_start="'"$part2_paragraph1_start"'" --index_end="'"$part2_paragraph1_end"'"'
    'python3 rq_9_stock_adj_factor.py --market=["XSHG","BJSE"] --index_start="'"$part2_paragraph2_start"'" --index_end="'"$part2_paragraph2_end"'"'
    'python3 rq_11_stock_daily_klines.py --market=["XSHE"] --index_start="'"$part1_paragraph1_start"'" --index_end="'"$part1_paragraph1_end"'"'
    'python3 rq_11_stock_daily_klines.py --market=["XSHE"] --index_start="'"$part1_paragraph2_start"'" --index_end="'"$part1_paragraph2_end"'"'
    'python3 rq_11_stock_daily_klines.py --market=["XSHG","BJSE"] --index_start="'"$part2_paragraph1_start"'" --index_end="'"$part2_paragraph1_end"'"'
    'python3 rq_11_stock_daily_klines.py --market=["XSHG","BJSE"] --index_start="'"$part2_paragraph2_start"'" --index_end="'"$part2_paragraph2_end"'"'
)


echo "=== Step3: 按并发数$MAX_JOBS，并发处理${#BATCH1_TASKS_LIST[@]} 个任务：index_components, industry_components, Stok_Klines, stock_adj, index_klines [$(date +"%Y-%m-%d %H:%M:%S")] ==="


# 启动初始的MAX_JOBS个任务
for ((i=0; i < MAX_JOBS && i < ${#BATCH1_TASKS_LIST[@]}; i++)); do
    echo "启动任务: ${BATCH1_TASKS_LIST[i]}"
    eval "${BATCH1_TASKS_LIST[i]}" &
done

# 记录下一个要启动的任务索引
next_index=$MAX_JOBS

# 循环处理剩余任务
while [ $next_index -lt ${#BATCH1_TASKS_LIST[@]} ]; do
    # 等待任意一个任务完成
    wait -n
    # 启动下一个任务
    echo "增加任务: ${BATCH1_TASKS_LIST[next_index]}"
    sleep 5
    eval "${BATCH1_TASKS_LIST[next_index]}" &
    ((next_index++))
done
wait


echo "=== Step4: 开始处理聚合数据 [$(date +"%Y-%m-%d %H:%M:%S")] ==="
python3 rq_5_industry_components_step2.py
python3 rq_16_stock_1m_aggregate.py
wait

echo "=== Data Download Batch1 Completed [$(date +"%Y-%m-%d %H:%M:%S")] ==="


### 等待进入batch2 #############################################################################################################

# 判断是否到20:05,进入batch2
current_seconds=$((10#$(date -d "$(date +%H:%M:%S)" +%s) - 10#$(date -d "00:00:00" +%s)))
target_seconds=$((10#$(date -d "20:05:00" +%s) - 10#$(date -d "00:00:00" +%s)))

# 判断当前时间是否早于20:05:00
if [ $current_seconds -lt $target_seconds ]; then
    # 计算需要sleep的秒数
    sleep_seconds=$((target_seconds - current_seconds))
    echo "当前时间早于20:00:00，需要等待 ${sleep_seconds} 秒"
    echo "等待至20:00:00..."
    # 使用sleep等待
    sleep $sleep_seconds
    
    echo "等待完成，当前时间: $(date '+%H:%M:%S')"
else
    echo "当前时间已晚于或等于20:00:00，直接执行后续任务"
fi

### 数据下载batch2【20:00】 #############################################################################################################

BATCH2_TASKS_LIST=(
    'python3 rq_31_stock_daily_factor.py --market=["XSHE"] --index_start="'"$part1_paragraph1_start"'" --index_end="'"$part1_paragraph1_end"'"'
    'python3 rq_31_stock_daily_factor.py --market=["XSHE"] --index_start="'"$part1_paragraph2_start"'" --index_end="'"$part1_paragraph2_end"'"'
    'python3 rq_31_stock_daily_factor.py --market=["XSHG","BJSE"] --index_start="'"$part2_paragraph1_start"'" --index_end="'"$part2_paragraph1_end"'"' 
    'python3 rq_31_stock_daily_factor.py --market=["XSHG","BJSE"] --index_start="'"$part2_paragraph2_start"'" --index_end="'"$part2_paragraph2_end"'"' 
    'python3 rq_21_stock_daily_turn.py --market=["XSHE"] --index_start="'"$part1_paragraph1_start"'" --index_end="'"$part1_paragraph1_end"'"'
    'python3 rq_21_stock_daily_turn.py --market=["XSHE"] --index_start="'"$part1_paragraph2_start"'" --index_end="'"$part1_paragraph2_end"'"' 
    'python3 rq_21_stock_daily_turn.py --market=["XSHG","BJSE"] --index_start="'"$part2_paragraph1_start"'" --index_end="'"$part2_paragraph1_end"'"' 
    'python3 rq_21_stock_daily_turn.py --market=["XSHG","BJSE"] --index_start="'"$part2_paragraph2_start"'" --index_end="'"$part2_paragraph2_end"'"'  
    'python3 rq_32_stock_daily_capital_flow.py --market=["XSHE"] --index_start="'"$part1_paragraph1_start"'" --index_end="'"$part1_paragraph1_end"'"'
    'python3 rq_32_stock_daily_capital_flow.py --market=["XSHE"] --index_start="'"$part1_paragraph2_start"'" --index_end="'"$part1_paragraph2_end"'"'      
    'python3 rq_32_stock_daily_capital_flow.py --market=["XSHG","BJSE"] --index_start="'"$part2_paragraph1_start"'" --index_end="'"$part2_paragraph1_end"'"' 
    'python3 rq_32_stock_daily_capital_flow.py --market=["XSHG","BJSE"] --index_start="'"$part2_paragraph2_start"'" --index_end="'"$part2_paragraph2_end"'"' 
)

echo "=== Step5: 按并发数$MAX_JOBS，并发处理${#BATCH2_TASKS_LIST[@]} 个任务：stock factor, turn, capital [$(date +"%Y-%m-%d %H:%M:%S")] ==="

# 启动初始的MAX_JOBS个任务
for ((i=0; i < MAX_JOBS && i < ${#BATCH2_TASKS_LIST[@]}; i++)); do
    echo "启动任务: ${BATCH2_TASKS_LIST[i]}"
    eval "${BATCH2_TASKS_LIST[i]}" &
done


# 记录下一个要启动的任务索引
next_index=$MAX_JOBS

# 循环处理剩余任务
while [ $next_index -lt ${#BATCH2_TASKS_LIST[@]} ]; do
    # 等待任意一个任务完成
    wait -n
    # 启动下一个任务
    echo "增加任务: ${BATCH2_TASKS_LIST[next_index]}"
    sleep 5
    eval "${BATCH2_TASKS_LIST[next_index]}" &
    ((next_index++))
done
wait

echo "=== Data Download Batch2 Completed [$(date +"%Y-%m-%d %H:%M:%S")] ==="


### 数据转化QLIB #############################################################################
# ----------------------------------------------------------------------------------------- #
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