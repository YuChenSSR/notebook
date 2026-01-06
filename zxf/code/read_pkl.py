import pickle
import numpy as np
import pandas as pd
data_dir = f"/home/idc2/notebook/zxf/data/master_results/csi800_20260105_f6_20150101_20251231"



### 2. 读取数据
with open(f'{data_dir}/csi800_self_dl_test.pkl', 'rb') as f:
    dl_train = pickle.load(f)

print(dl_train.data.shape)

for col in dl_train.data.columns:
    print(col)

### 3. 转化筛选打印
# df = dl_train.data
# df = df.droplevel(0, axis=1)
# df.reset_index(inplace=True)
# df = df[df['instrument'] == 'SH600588']
# df = df[['datetime', 'instrument', 'Ref($adjclose,-5)/Ref($adjclose,-1)-1 - Mask(Ref($adjclose,-5)/Ref($adjclose,-1)-1, "sh000906")']]
# df = df[(df['datetime'] >= '2025-06-01') & (df['datetime'] <= '2025-06-30')]

# # print(df.tail(20).to_string(max_cols=None))
# print(df)




# ### 4. 查询NaN值
# df = dl_train.data.copy()
# # 找到所有NaN值的位置
# nan_locations = np.where(pd.isna(df))
# # 获取行索引和列索引
# rows = nan_locations[0]
# print(f"在DataFrame中共找到 {len(rows)} 个NaN值")
