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
        pred_filename=pred_filename,
        backtest_start_date="2025-09-01",
    )    
    return account_detail_s

def main(
    market_name: str = "csi800",
    folder_name: str = "csi800_20251211_20150101_20251208",
    data_path: str = f"/home/idc2/notebook/zxf/data/modoel_training",
    bt_n: int = 5,
):
    expt_path = f"{data_path}/{folder_name}"
    output_path = f"{expt_path}/Backtest_Results/my3"
    os.makedirs(output_path, exist_ok=True)

    backday=8
    backtest_my_results = pd.DataFrame()
    
    ### 读取qlib回测结果，按master实验参数集合，取年化最大的bt_n个种子
    try:
        qlib_backtest_results_filename = f"{expt_path}/Backtest_Results/qlib/backtest_qlib_results.csv"
        qlib_bt_r = pd.read_csv(qlib_backtest_results_filename)
    except Exception as e:
        logger.error(f"Qlib backtest results get failed: {str(e)}")
        sys.exit(1)

    top_per_group = qlib_bt_r.groupby(['Seed', 'D_model', 'LR', 'Dropout']).apply(lambda x: x.nlargest(bt_n, 'annual_return')).reset_index(drop=True)


    
    for index, row in top_per_group.iterrows():
        seed = int(row['Seed'])
        d_model = int(row['D_model'])
        lr = row['LR']
        dropout = row['Dropout']
        step = int(row['Step'])

        pred_filename = f"{expt_path}/Predictions/master_predictions_backday_{backday}_{market_name}_{seed}_{d_model}_{lr}_{dropout}_{step}.csv"

        print("\n" + "-" * 100)
        print(f"Seed: {seed} - D_model: {d_model} - LR: {lr} - Dropout: {dropout} - Step: {step}")

        ### 2. 按1的结果配置回测参数进行回测
        # top_k_list=[30, 20, 10, 5]
        # n_drop_ratios=[1, 0.8, 0.5, 0.3, 0.1]
        # hold_p_list=[5, 4, 3, 2, 1]

        top_k_list=[30, 10]
        n_drop_ratios=[1, 0.1]
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
        for index, row in params.iterrows():
            top_k = row['top_k']
            n_drop = row['n_drop']
            hold_p = row['hold_p']
            print(f"\ntop_k:{top_k} - n_drop:{n_drop} - hold_p:{hold_p}")

            backtest = Backtest(
                top_k=top_k,
                n_drop=n_drop,
                hold_p=hold_p,        
                pred_filename=pred_filename
            )
            account_detail = backtest.run()
            account_detail.to_csv(f'{output_path}/backtest_my_{seed}_{d_model}_{lr}_{dropout}_{step}_{top_k}_{n_drop}_{hold_p}.csv', index=False, date_format='%Y-%m-%d')
            
            account_detail_s = account_detail[-1:]
            account_detail_s['Seed'] = seed
            account_detail_s['D_model'] = d_model
            account_detail_s['LR'] = lr
            account_detail_s['Dropout'] = dropout
            account_detail_s['Step'] = step
            
            account_detail_s['top_k'] = top_k
            account_detail_s['n_drop'] = n_drop
            account_detail_s['hold_p'] = hold_p

            account_detail_s = account_detail_s[
                ['Seed','D_model','LR','Dropout','Step','top_k','n_drop','hold_p',
                'total_return','annual_return','max_drawdown','annual_turnover','volatility',
                 'sharpe','information_ratio','alpha','beta','annual_benchmark_return','winning_rate_cumulant']]
            
            print(account_detail_s.to_string(max_cols=None))
            backtest_my_results = pd.concat([backtest_my_results, account_detail_s],ignore_index=True)
    backtest_my_results.to_csv(f'{output_path}/backtest_my_results_new1.csv', index=False, date_format='%Y-%m-%d')

    
    df_qlib = qlib_bt_r.copy()
    df_qlib = df_qlib.rename(columns={
        'annual_return': 'qlib_annual_return',
        'annual_excess_return': 'qlib_annual_excess_return',
        'max_drawdown': 'qlib_max_drawdown',
        'excess_max_drawdown': 'qlib_excess_max_drawdown',
        'sharpe': 'qlib_sharpe',
        'information_ratio': 'qlib_information_ratio',
    })
    df_results = pd.merge(backtest_my_results, df_qlib, on=['Seed', 'D_model', 'LR', 'Dropout', 'Step'], how='left')
    df_results.to_csv(f'{expt_path}/Backtest_Results/Backtest_Results_new3.csv', index=False, date_format='%Y-%m-%d')


    



if __name__ == "__main__":
    fire.Fire(main)