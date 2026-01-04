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
# echo "part1_paragraph1_start: $part1_paragraph1_start"
# echo "part1_paragraph1_end: $part1_paragraph1_end"
# echo "part1_paragraph2_start: $part1_paragraph2_start"
# echo "part1_paragraph2_end: $part1_paragraph2_end"
# echo "part2_paragraph1_start: $part2_paragraph1_start"
# echo "part2_paragraph1_end: $part2_paragraph1_end"
# echo "part2_paragraph2_start: $part2_paragraph2_start"
# echo "part2_paragraph2_end: $part2_paragraph2_end"


echo "=== Step2: calendar, stock列表, index列表, industry列表 [$(date +"%Y-%m-%d %H:%M:%S")] ==="
# python3 rq_0_trade_date.py &
python3 rq_1_stock_list.py &
python3 rq_2_index_list.py &
python3 rq_3_industry_list.py &
wait


echo "=== Step3: index_components, industry_components [$(date +"%Y-%m-%d %H:%M:%S")] ==="
python3 rq_4_index_components.py &
python3 rq_12_index_daily_klines.py &
python3 rq_5_industry_components_step1.py &
wait

echo "=== Step4: Klines, stock_adj, index_klines [$(date +"%Y-%m-%d %H:%M:%S")] ==="
python3 rq_9_stock_adj_factor.py --market=["XSHE"] &
python3 rq_9_stock_adj_factor.py --market=["XSHG","BJSE"] &
python3 rq_11_stock_daily_klines.py --market=["XSHE"] &
python3 rq_11_stock_daily_klines.py --market=["XSHG","BJSE"] &
wait


echo "=== Step5: klines_1m, industry_components [$(date +"%Y-%m-%d %H:%M:%S")] ==="
python3 rq_15_stock_1m_klines.py --market=["XSHE"] --index_start="$part1_paragraph1_start" --index_end="$part1_paragraph1_end" &
python3 rq_15_stock_1m_klines.py --market=["XSHE"] --index_start="$part1_paragraph2_start" --index_end="$part1_paragraph2_end" &

python3 rq_15_stock_1m_klines.py --market=["XSHG","BJSE"] --index_start="$part2_paragraph1_start" --index_end="$part2_paragraph1_end" &
python3 rq_15_stock_1m_klines.py --market=["XSHG","BJSE"] --index_start="$part2_paragraph2_start" --index_end="$part2_paragraph2_end" &
python3 rq_5_industry_components_step2.py &
wait

echo "=== Step6: 开始处理聚合数据 [$(date +"%Y-%m-%d %H:%M:%S")] ==="
python3 rq_16_stock_1m_aggregate.py
wait

echo "=== Data Download Batch1 Completed [$(date +"%Y-%m-%d %H:%M:%S")] ==="