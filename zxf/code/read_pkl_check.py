import pickle
import fire
import numpy as np
import pandas as pd
from loguru import logger


def read_data(data_dir):
    with open(f'{data_dir}/csi800_self_dl_train.pkl', 'rb') as f:
        dl_train = pickle.load(f)
    return dl_train

def data_check(dl_train):
    ### 数据检查
    df = dl_train.data.copy()
    
    ### 总值检查
    # 总值
    total_values = df.size
    # 总NaN值数量
    nan_count = df.isna().sum().sum()
    nan_ratio = nan_count / total_values
    # 总0值数量
    zero_count = (df == 0).sum().sum()
    zero_ratio = zero_count / total_values
    
    print("=== 整体统计 ===")
    print(f"总值数量: {total_values}")
    print(f"1) 'NaN'数量: {nan_count}, 比例: {nan_ratio:.4f}")
    print(f"2) '0'值数量: {zero_count}, 比例: {zero_ratio:.4f}")
    
    
    ### 列值检查
    # 每列总值数量（含 NaN）
    col_total = df.count() + df.isna().sum()    # 或直接 df.shape[0]
    col_nan = df.isna().sum()
    col_nan_ratio = col_nan / col_total
    col_zero = (df == 0).sum()
    col_zero_ratio = col_zero / col_total
    
    result = pd.DataFrame({
        "total_values": col_total,
        "nan_count": col_nan,
        "nan_ratio": col_nan_ratio,
        "zero_count": col_zero,
        "zero_ratio": col_zero_ratio,
    })
    
    result.columns = result.columns.get_level_values(-1)
    result = result.reset_index()
    # result.to_csv('./features_check.csv', index=False, date_format='%Y-%m-%d')
    
    print("=== 每列统计 ===")
    result = result[result['zero_ratio'] >= 0.05]
    
    
    print(result.to_string(max_cols=None))
    print(len(result))

def data_details_check(dl_train):
    df = dl_train.data.copy()

    df.columns = df.columns.get_level_values(-1)
    df = df.reset_index()

    df_check = df[df['$open/($close+1e-12)']==0]

    df_check = df_check[['datetime', 'instrument', '$open/($close+1e-12)']]
    
    print(df_check)
    

def main(
    data_dir: str = "/home/idc2/notebook/zxf/data/master_results/csi800_20251130_20150101_20251128",
):
    ### 读取数据
    dl_train = read_data(data_dir)

    ### 整体检查NaN、0
    # data_check(dl_train)

    ### 按项目检查
    data_details_check(dl_train)
    


if __name__ == "__main__":
    fire.Fire(main)





