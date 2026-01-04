from qlib import init
from qlib.constant import REG_CN
import pandas as pd
import os


### 初始化
qlib_data_path = '/home/idc2/notebook/qlib_bin/cn_data_train'
init(provider_uri=qlib_data_path, region=REG_CN)
from qlib.data import D


### 读取列表
instruments_dict = D.list_instruments(D.instruments("csi800b"))
instruments = list(instruments_dict.keys())


### 读取数据
# instruments = ["SH600588", "SZ302132"]
fields = ["$open", "$high", "$low", "$close", "$volume"]
df = D.features(
    instruments,
    fields,
    start_time="2015-01-01", end_time="2025-11-25"
)
df = df.reset_index()
df_data = df[['datetime']].drop_duplicates()
first_date = df['datetime'].min()

date_number = len(df_data)

i = 0
for symbol in instruments:
    df_temp=df[df['instrument']==symbol]
    temp_first_date = df_temp['datetime'].min()


    # 判断数据长度
    if temp_first_date <= first_date :
        # 更正列名
        for col in df_temp.columns:
            col_clean = col.replace('$', '').title()
            if col in ["instrument", "datetime"]:
                continue
            df_temp = df_temp.rename(columns={col: f"{col_clean}_{symbol}"})
    
        df_temp = df_temp.drop(['instrument'], axis=1)
        df_data = pd.merge(df_data, df_temp, on="datetime", how='left')
        i += 1
        print(f"{symbol} 已合并")
    else:
        print(f"{symbol} 长度不够")

# 填充
for col in df_data.columns:  
    if col == "datetime":
        continue
    df_data[col] = df_data[col].fillna(method="ffill").fillna(method="bfill").fillna(1e-12)

# 列排序
order = ['Close', 'High', 'Low', 'Open', 'Volume']
cols = df_data.columns.tolist()
first_col = ['datetime'] if 'datetime' in cols else []
other_cols = [c for c in cols if c != 'datetime']
sorted_cols = first_col + [
    c
    for feature in order
    for c in other_cols
    if feature.lower() in c.lower()  # 特征名不区分大小写
]
remaining = [c for c in other_cols if c not in sorted_cols]
sorted_cols += remaining
df_data = df_data[sorted_cols]

# 保存
df_data = df_data.rename(columns={'datetime': 'date'})
df_data.to_csv("u_cast.csv", index=False, date_format='%Y-%m-%d')
print(f"completed:{i}")

# print(df_data.tail(10).to_string(max_cols=None))

