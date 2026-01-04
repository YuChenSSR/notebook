 #!/bin/bash

set -e
set -x

# 切换虚拟环境
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/opt/anaconda3/bin

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate pytorch

# rqdatac license
# license_luo.txt, license_hao.txt, license_lin.txt
# LICENSE_FILE="/home/idc2/notebook/rqdata/.rqdatac/license.txt"
LICENSE_FILE="/home/idc2/notebook/rqdata/cron/license_hao.txt"

END_DATE="2025-12-12"
INDEX_START=350
INDEX_END=$((INDEX_START + 350))
# 5550

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
CODE_PATH="$HOME/notebook/futures/code/data_download"
cd "$CODE_PATH"

echo "=== Futures Klines 1M ==="
python3 rq_f11_futures_klines_1m.py --end_date="$END_DATE" # --index_start="$INDEX_START" --index_end="$INDEX_END"

echo "=== Print License Info ==="
rqsdk license info
