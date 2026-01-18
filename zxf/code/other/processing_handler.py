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



def main(
    data_dir: str = "/home/idc2/notebook/zxf/data/master_results/csi800_20260117_20150101_20260115",
    market_name: str = 'csi800',
):
    ### 定义目录
    data_path = f"{data_dir}"

    # 门控
    gate_cols = [
        '$i_change_1',
        '$i_change_1_mean_5',
        '$i_change_1_std_5',
        '$i_change_1_mean_10',
        '$i_change_1_std_10',
        '$i_change_1_mean_20',
        '$i_change_1_std_20',
        '$i_change_1_mean_30',
        '$i_change_1_std_30',
        '$i_change_1_mean_60',
        '$i_change_1_std_60',
    ]
    ### 读取需要的列名

    filter_cols_path = f"{data_path}/factor_analysis_report_20260117_163308.csv"
    filter_cols_file = pd.read_csv(filter_cols_path)

    # 筛选条件
    df = filter_cols_file.copy()
    con = df['final_advice'] == "KEEP_CORE"
    df_s = df[con]

    filte_cols = sorted(df_s['factor'].tolist())


    need_gate_cols = [x for x in gate_cols if x not in filte_cols]
    need_cols = filte_cols + need_gate_cols
    need_cols = list(dict.fromkeys(need_cols))

    
    for col in need_cols:
        col = '"' + col + '",'
        print(col)




if __name__ == "__main__":
    fire.Fire(main)
    


    

