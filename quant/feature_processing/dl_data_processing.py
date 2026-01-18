import pickle
import fire
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.cluster import AgglomerativeClustering
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm
from pathlib import Path

from pandas import IndexSlice as idx

# 数据读取
def read_pkl(data_dir):
    with open(data_dir, "rb") as f:
        dl = pickle.load(f)
    return dl

def filter_column(dl_data_path,save_path, need_cols):
    
    dl_data = read_pkl(dl_data_path)
    df = dl_data.data

    # 筛选列
    feature_names = df['feature'].columns
    feature_df = df.loc[:, ('feature', need_cols)]
    
    label_df = df.loc[:, 'label']
    label_df.columns = pd.MultiIndex.from_product(
        [['label'], label_df.columns]
    )
    
    df_filtered = pd.concat([feature_df, label_df], axis=1)
    df_filtered = df_filtered.loc[:, df.columns.intersection(df_filtered.columns)]
    dl_data.data = df_filtered

    # 保存
    with open(save_path, 'wb') as file: pickle.dump(dl_data, file)
            
    return dl_data

def main(
    data_dir: str = "/home/a/notebook/zxf/data/Daily_data/Good_seed/seed7",
    market_name: str = 'csi800',
):
    ### 定义目录
    # data_dir = "/home/a/notebook/zxf/data/Daily_data/Training_data/csi800"
    # data_path = f"{data_dir}/{market_name}"
    data_path = f"{data_dir}"

    # 门控
    gate_cols = [
        # '$adjusted_fully_diluted_earnings_per_share_ttm',
        # '$adjusted_return_on_equity_ttm',
        # '$return_on_asset_net_profit_ttm',
        # '$net_profit_margin_ttm',
        # '$net_profit_to_revenue_ttm',
        # '$total_asset_turnover_ttm',
        # '$operating_cash_flow_per_share_ttm',
        # '$ocf_to_debt_ttm',
        # '$debt_to_asset_ratio_ttm',
        # '$debt_to_equity_ratio_ttm',
        # '$book_value_per_share_ttm',
        # '$inc_revenue_ttm',
        # '$inc_book_per_share_ttm',
        # '$net_profit_growth_ratio_ttm',
        '$i_change_1',
        '$i_change_1_mean_5',
        '$i_change_1_std_5',
        # '$i_change_1_mean_10',
        # '$i_change_1_std_10',
        '$i_change_1_mean_20',
        '$i_change_1_std_20',
        # '$i_change_1_mean_30',
        # '$i_change_1_std_30',
        # '$i_change_1_mean_60',
        # '$i_change_1_std_60',
        
        'Mask($close/Ref($close,1)-1, "sh000001")',
        'Mask(Mean($close/Ref($close,1)-1,5), "sh000001")',
        'Mask(Std($close/Ref($close,1)-1,5), "sh000001")',
        'Mask(Mean($volume,5)/$volume, "sh000001")',
        'Mask(Std($volume,5)/$volume, "sh000001")',
        # 'Mask(Mean($close/Ref($close,1)-1,10), "sh000001")',
        # 'Mask(Std($close/Ref($close,1)-1,10), "sh000001")',
        # 'Mask(Mean($volume,10)/$volume, "sh000001")',
        # 'Mask(Std($volume,10)/$volume, "sh000001")',
        'Mask(Mean($close/Ref($close,1)-1,20), "sh000001")',
        'Mask(Std($close/Ref($close,1)-1,20), "sh000001")',
        'Mask(Mean($volume,20)/$volume, "sh000001")',
        'Mask(Std($volume,20)/$volume, "sh000001")',
        # 'Mask(Mean($close/Ref($close,1)-1,30), "sh000001")',
        # 'Mask(Std($close/Ref($close,1)-1,30), "sh000001")',
        # 'Mask(Mean($volume,30)/$volume, "sh000001")',
        # 'Mask(Std($volume,30)/$volume, "sh000001")',
        # 'Mask(Mean($close/Ref($close,1)-1,60), "sh000001")',
        # 'Mask(Std($close/Ref($close,1)-1,60), "sh000001")',
        # 'Mask(Mean($volume,60)/$volume, "sh000001")',
        # 'Mask(Std($volume,60)/$volume, "sh000001")',
        
        'Mask($close/Ref($close,1)-1, "sh000300")',
        'Mask(Mean($close/Ref($close,1)-1,5), "sh000300")',
        'Mask(Std($close/Ref($close,1)-1,5), "sh000300")',
        'Mask(Mean($volume,5)/$volume, "sh000300")',
        'Mask(Std($volume,5)/$volume, "sh000300")',
        # 'Mask(Mean($close/Ref($close,1)-1,10), "sh000300")',
        # 'Mask(Std($close/Ref($close,1)-1,10), "sh000300")',
        # 'Mask(Mean($volume,10)/$volume, "sh000300")',
        # 'Mask(Std($volume,10)/$volume, "sh000300")',
        'Mask(Mean($close/Ref($close,1)-1,20), "sh000300")',
        'Mask(Std($close/Ref($close,1)-1,20), "sh000300")',
        'Mask(Mean($volume,20)/$volume, "sh000300")',
        'Mask(Std($volume,20)/$volume, "sh000300")',
        # 'Mask(Mean($close/Ref($close,1)-1,30), "sh000300")',
        # 'Mask(Std($close/Ref($close,1)-1,30), "sh000300")',
        # 'Mask(Mean($volume,30)/$volume, "sh000300")',
        # 'Mask(Std($volume,30)/$volume, "sh000300")',
        # 'Mask(Mean($close/Ref($close,1)-1,60), "sh000300")',
        # 'Mask(Std($close/Ref($close,1)-1,60), "sh000300")',
        # 'Mask(Mean($volume,60)/$volume, "sh000300")',
        # 'Mask(Std($volume,60)/$volume, "sh000300")',
        
        'Mask($close/Ref($close,1)-1, "sh000905")',
        'Mask(Mean($close/Ref($close,1)-1,5), "sh000905")',
        'Mask(Std($close/Ref($close,1)-1,5), "sh000905")',
        'Mask(Mean($volume,5)/$volume, "sh000905")',
        'Mask(Std($volume,5)/$volume, "sh000905")',
        # 'Mask(Mean($close/Ref($close,1)-1,10), "sh000905")',
        # 'Mask(Std($close/Ref($close,1)-1,10), "sh000905")',
        # 'Mask(Mean($volume,10)/$volume, "sh000905")',
        # 'Mask(Std($volume,10)/$volume, "sh000905")',
        'Mask(Mean($close/Ref($close,1)-1,20), "sh000905")',
        'Mask(Std($close/Ref($close,1)-1,20), "sh000905")',
        'Mask(Mean($volume,20)/$volume, "sh000905")',
        'Mask(Std($volume,20)/$volume, "sh000905")',
        # 'Mask(Mean($close/Ref($close,1)-1,30), "sh000905")',
        # 'Mask(Std($close/Ref($close,1)-1,30), "sh000905")',
        # 'Mask(Mean($volume,30)/$volume, "sh000905")',
        # 'Mask(Std($volume,30)/$volume, "sh000905")',
        # 'Mask(Mean($close/Ref($close,1)-1,60), "sh000905")',
        # 'Mask(Std($close/Ref($close,1)-1,60), "sh000905")',
        # 'Mask(Mean($volume,60)/$volume, "sh000905")',
        # 'Mask(Std($volume,60)/$volume, "sh000905")',
        
        'Mask($close/Ref($close,1)-1, "sh000906")',
        'Mask(Mean($close/Ref($close,1)-1,5), "sh000906")',
        'Mask(Std($close/Ref($close,1)-1,5), "sh000906")',
        'Mask(Mean($volume,5)/$volume, "sh000906")',
        'Mask(Std($volume,5)/$volume, "sh000906")',
        # 'Mask(Mean($close/Ref($close,1)-1,10), "sh000906")',
        # 'Mask(Std($close/Ref($close,1)-1,10), "sh000906")',
        # 'Mask(Mean($volume,10)/$volume, "sh000906")',
        # 'Mask(Std($volume,10)/$volume, "sh000906")',
        'Mask(Mean($close/Ref($close,1)-1,20), "sh000906")',
        'Mask(Std($close/Ref($close,1)-1,20), "sh000906")',
        'Mask(Mean($volume,20)/$volume, "sh000906")',
        'Mask(Std($volume,20)/$volume, "sh000906")',
        # 'Mask(Mean($close/Ref($close,1)-1,30), "sh000906")',
        # 'Mask(Std($close/Ref($close,1)-1,30), "sh000906")',
        # 'Mask(Mean($volume,30)/$volume, "sh000906")',
        # 'Mask(Std($volume,30)/$volume, "sh000906")',
        # 'Mask(Mean($close/Ref($close,1)-1,60), "sh000906")',
        # 'Mask(Std($close/Ref($close,1)-1,60), "sh000906")',
        # 'Mask(Mean($volume,60)/$volume, "sh000906")',
        # 'Mask(Std($volume,60)/$volume, "sh000906")'
    ]

    
    ### 读取需要的列名
    # notebook/zxf/data/Daily_data/Good_seed/seed7/factor_analysis_report_20260107_174156.csv
    
    filter_cols_path = f"{data_path}/factor_analysis_report_20260107_174156.csv"
    filter_cols_file = pd.read_csv(filter_cols_path)

    # 筛选条件
    df = filter_cols_file.copy()
    con = df['final_advice'] == "KEEP_CORE"
    df_s = df[con]

    filte_cols = df_s['factor'].tolist()
    need_cols = filte_cols + gate_cols
    need_cols = list(dict.fromkeys(need_cols))


    # dl_type = ['test']
    dl_type = ['train', 'valid', 'test']
    for d_type in dl_type:
        dl_data_path_raw = Path(f"{data_path}/{market_name}_self_dl_{d_type}.pkl")
        dl_data_path_new = Path(f"{data_path}/{market_name}_self_dl_{d_type}_raw.pkl")

        if dl_data_path_raw.exists():
            dl_data_path_raw.rename(dl_data_path_new)
        

        # dl_data_path = f"{data_path}/{market_name}_self_dl_{d_type}.pkl"
        # save_path = f"{data_path}/{market_name}_self_dl_{d_type}_8.pkl"

        save_path = dl_data_path_raw
        dl_data = filter_column(dl_data_path_new, save_path, need_cols)
        # print(dl_data.data)



if __name__ == "__main__":
    fire.Fire(main)
    


    

