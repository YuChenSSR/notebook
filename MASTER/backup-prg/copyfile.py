import shutil
import os

# 源目录
source_dir = "/MASTER/data/self_exp/opensource"
# 目标目录
target_dir = "//master_new/data"

# 复制整个目录（包括子目录）
shutil.copytree(source_dir, target_dir)