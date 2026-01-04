import pandas as pd
import numpy as np
from loguru import logger
import fire
import sys

def read_results(results_path):
    try:
        return pd.read_csv(results_path)
    except Exception as e:
        sys.exit(1)

def results_analysis(df_results):
    df = df_results.copy()

    
    # 0 <= 超额收益 < 0.1
    # con_1_1 = df['annual_excess_return'] >= 0
    # con_1_2 = df['annual_excess_return'] < 0.10
    # con_1 = con_1_1 & con_1_2
    # df_temp1 = df[con_1]
    # total_count = len(df_temp1)
    # print(total_count1)
    # count_result1 = df_temp1.groupby(['D_model', 'LR', 'Dropout']).size().reset_index(name='count')
    # count_result1 = count_result1.sort_values(by=['D_model', 'LR', 'Dropout'])

    # 0.1 <= 超额收益 < 0.2
    # con_2_1 = df['annual_excess_return'] >= 0.1
    # con_2_2 = df['annual_excess_return'] < 0.20
    # con_2 = con_2_1 & con_2_2
    # df_temp2 = df[con_2]
    # total_count2 = len(df_temp2)
    # print(total_count2)
    # count_result2 = df_temp2.groupby(['D_model', 'LR', 'Dropout']).size().reset_index(name='count')
    # count_result2 = count_result2.sort_values(by=['D_model', 'LR', 'Dropout'])

    # 0.2 <= 超额收益 < 0.3
    con_3_1 = df['annual_excess_return'] >= 0.2
    con_3_2 = df['annual_excess_return'] < 0.3
    con_3 = con_3_1 & con_3_2
    df_temp3 = df[con_3]
    total_count3 = len(df_temp3)
    print(total_count3)
    count_result3 = df_temp3.groupby(['D_model', 'LR', 'Dropout']).size().reset_index(name='count')
    count_result3 = count_result3.sort_values(by=['D_model', 'LR', 'Dropout'])
    print(count_result3)

    # 0.3 <= 超额收益
    con_4_1 = df['annual_excess_return'] >= 0.3
    con_4_2 = df['annual_excess_return'] < 1
    con_4 = con_4_1 & con_4_2
    df_temp4 = df[con_4]
    total_count4 = len(df_temp4)
    print(total_count4)
    count_result4 = df_temp4.groupby(['D_model', 'LR', 'Dropout']).size().reset_index(name='count')
    count_result4 = count_result4.sort_values(by=['D_model', 'LR', 'Dropout'])
    
    print(count_result4)
    


def main(
    data_path: str = "/home/idc2/notebook/zxf/data/modoel_training/csi800_20251211_20150101_20251208/Backtest_Results/qlib/backtest_qlib_results.csv"
):
    df_results = read_results(data_path)
    results_analysis(df_results)
    
    # print(df_results)





if __name__ == "__main__":
    fire.Fire(main)