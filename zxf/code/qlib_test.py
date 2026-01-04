from qlib import init
from qlib.constant import REG_CN
import os


# 配置路径
qlib_data_path = '/home/idc2/notebook/qlib_bin/cn_data_train'


# 初始化
init(provider_uri=qlib_data_path, region=REG_CN)
from qlib.data import D

instruments = ["SH600588"]
fields = [
    "$adjclose",
    "Ref($adjclose,-5)/Ref($adjclose,-1)-1",
    "Std(Ref($adjclose, -1)/$adjclose-1,5)",
    # '(Ref($adjclose,-5)/Ref($adjclose,-1)-1)/(Std(Ref($adjclose, -1)/Ref($adjclose,0)-1,5)+1e-6)',
    # 'Ref($adjclose,-5)/Ref($adjclose,-1)-1 - Mask(Ref($adjclose,-5)/Ref($adjclose,-1)-1, "sh000906")',


    "(Ref($adjclose,-5)/Ref($adjclose,-1)-1) / (1 + Std(Ref($adjclose,-1)/$adjclose-1, 5))",
    "(Ref($adjclose,-5)/Ref($adjclose,-1)-1) / (1 + Power(Std(Ref($adjclose,-5)/Ref($adjclose,-4), Ref($adjclose,-4)/Ref($adjclose,-3), Ref($adjclose,-3)/Ref($adjclose,-2), Ref($adjclose,-2)/Ref($adjclose,-1), Ref($adjclose,-1)/$adjclose),2))",

    # "If((Ref($adjclose,-5)/Ref($adjclose,-1)-1) >= 0,(Ref($adjclose,-5)/Ref($adjclose,-1)-1) / (1 + 20 * Std(Abs(Ref($adjclose,-1)/Ref($adjclose,0)-1), 5)),(Ref($adjclose,-5)/Ref($adjclose,-1)-1) * (1 + 20 * Std(Abs(Ref($adjclose,-1)/Ref($adjclose,0)-1), 5)))",
    
    # "If((Ref($adjclose,-5)/Ref($adjclose,-1)-1) >= 0,(Ref($adjclose,-5)/Ref($adjclose,-1)-1) / (1 + Std(Ref($adjclose,-1)/$adjclose-1, 5)),(Ref($adjclose,-5)/Ref($adjclose,-1)-1) * (1 + Std(Ref($adjclose,-1)/$adjclose-1, 5)))",
    # # 请使用这个公式进行接下来的所有实验
    # # "(Ref($adjclose,-5)/Ref($adjclose,-1)-1) / (1 + Std($adjclose/Ref($adjclose,1)-1, 5))",

    # "If((Ref($adjclose,-5)/Ref($adjclose,-1)-1) >= 0,(Ref($adjclose,-5)/Ref($adjclose,-1)-1) / (1 + 10 * Std(Ref($adjclose,-1)/$adjclose-1, 5)),(Ref($adjclose,-5)/Ref($adjclose,-1)-1) / (1 - 10 * Std(Ref($adjclose,-1)/$adjclose-1, 5)))"
]

features = D.features(
    instruments,
    fields,
    start_time="2025-06-01", end_time="2025-06-30")
print(features.to_string(max_cols=None))
