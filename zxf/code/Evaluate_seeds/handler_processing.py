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
    data_dir: str = "/home/idc2/notebook/zxf/code/Evaluate_seeds",
    market_name: str = 'csi800',
):
    ### 定义目录
    data_path = f"{data_dir}"



    
    ### 读取需要的列名
    
    filter_cols_path = f"{data_path}/factor_analysis_report_20260114_130111.csv"
    filter_cols_file = pd.read_csv(filter_cols_path)

    # 筛选条件
    df = filter_cols_file.copy()
    con = df['final_advice'] == "KEEP_CORE"
    df_s = df[con]

    filte_cols = df_s['factor'].tolist()
    filte_cols = sorted(filte_cols)

    for col in filte_cols:
        col = '"' + col + '",'
        print(col)

        
    # need_cols = filte_cols + gate_cols
    # need_cols = list(dict.fromkeys(need_cols))


    # # dl_type = ['test']
    # dl_type = ['train', 'valid', 'test']
    # for d_type in dl_type:
    #     dl_data_path_raw = Path(f"{data_path}/{market_name}_self_dl_{d_type}.pkl")
    #     dl_data_path_new = Path(f"{data_path}/{market_name}_self_dl_{d_type}_raw.pkl")

    #     if dl_data_path_raw.exists():
    #         dl_data_path_raw.rename(dl_data_path_new)
        

    #     # dl_data_path = f"{data_path}/{market_name}_self_dl_{d_type}.pkl"
    #     # save_path = f"{data_path}/{market_name}_self_dl_{d_type}_8.pkl"

    #     save_path = dl_data_path_raw
    #     dl_data = filter_column(dl_data_path_new, save_path, need_cols)
        # print(dl_data.data)



if __name__ == "__main__":
    fire.Fire(main)
    


    

