from qlib import init
from qlib.constant import REG_CN
import os


### 初始化
qlib_data_path = '/home/a/notebook/cn_data_train'
init(provider_uri=qlib_data_path, region=REG_CN)

from qlib.data import D

### 检查数据
instruments = ["SH601898"]
fields = [
    "Ref($d_low_time_frac, 10)/($d_low_time_frac+1e-12)"
]


features = D.features(
    instruments,
    fields,
    start_time="2025-12-01", end_time="2025-12-08")
print(features.to_string(max_cols=None))

