import pandas as pd
import fire
import sys
import os
from loguru import logger

from My_backtest import Backtest

def run_my_backtest(top_k, n_drop, hold_p, pred_filename):
    backtest = Backtest(
        top_k=top_k,
        n_drop=n_drop,
        hold_p=hold_p,        
        pred_filename=pred_filename
    )    
    return account_detail_s

def main(
    market_name: str = "csi800",
    folder_name: str = "csi800_20251211_20150101_20251208",
    data_path: str = f"/home/idc2/notebook/zxf/data/modoel_training",
    bt_n: int = 5,
):
    # expt_path = f"{data_path}/{folder_name}"
    # output_path = f"{expt_path}/Backtest_Results/my"
    # os.makedirs(output_path, exist_ok=True)

    # backday=8
    # backtest_my_results = pd.DataFrame()
    
    # ### 读取qlib回测结果，按master实验参数集合，取年化最大的bt_n个种子
    # try:
    #     qlib_backtest_results_filename = f"{expt_path}/Backtest_Results/qlib/backtest_qlib_results.csv"
    #     qlib_bt_r = pd.read_csv(qlib_backtest_results_filename)
    # except Exception as e:
    #     logger.error(f"Qlib backtest results get failed: {str(e)}")
    #     sys.exit(1)


    # top_per_group = qlib_bt_r.groupby(['Seed', 'D_model', 'LR', 'Dropout']).apply(lambda x: x.nlargest(bt_n, 'annual_return')).reset_index(drop=True)

    # top_per_group = top_per_group[['Seed', 'D_model', 'LR', 'Dropout', 'Step', 'annual_return', 'annual_excess_return']]
    
    # print(top_per_group.to_string(max_cols=None))
    # print(len(top_per_group))

    # top_per_group_z = top_per_group[top_per_group['annual_excess_return'] > 0]
    # print(len(top_per_group_z))




    top_k_list=[30, 50, 80]
    n_drop_ratios=[1, 0.5, 0.1]
    hold_p_list=[5, 1]

    params_c = [
        {'top_k': top_k, 'n_drop': max(1, int(top_k * ratio)), 'hold_p': hold_p} #, 'n_drop_ratio': ratio}
        for top_k in top_k_list
        for ratio in n_drop_ratios
        for hold_p in hold_p_list
        if int(top_k * ratio) <= top_k
    ]
    param_df = pd.DataFrame(params_c)
    params = param_df.drop_duplicates(keep='first').reset_index(drop=True)    # 去重
    print(params)
    

    
    # for index, row in top_per_group.iterrows():
    #     seed = int(row['Seed'])
    #     d_model = int(row['D_model'])
    #     lr = row['LR']
    #     dropout = row['Dropout']
    #     step = int(row['Step'])

  



if __name__ == "__main__":
    fire.Fire(main)