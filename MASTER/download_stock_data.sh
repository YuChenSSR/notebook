#!/bin/bash
set -e

# 获取日期参数
today=$1
echo $today

# 人工指定日期
# today="2025-04-02"
# echo ${today}

# 动态生成 URL
url="https://github.com/chenditc/investment_data/releases/download/${today}/qlib_bin.tar.gz"

# 删除上次压缩包
if [ -f ~/notebook/qlib_bin.tar.gz ]; then
    rm ~/notebook/qlib_bin.tar.gz
    echo "旧文件存在，已删除"
else
    echo "旧文件不存在，无需删除"
fi

# 下载文件到指定目录
wget -P ~/notebook "$url"

echo "文件已下载到 ~/notebook"

tar -xzvf ~/notebook/qlib_bin.tar.gz -C ~/notebook

echo "qlib_bin.tar.gz 已完成解压"