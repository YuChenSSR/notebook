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

def results_analysis(df_results, data_path):
    save_path = data_path
    
    df = df_results.copy()
    df['excess_return'] = df['annual_return'] - df['annual_benchmark_return']

    print(df.tail(1).to_string(max_cols=None))

    ### top_k、n_drop、hold_p 单个分析 —————————————————————————————————————————————————————————————————————————————————————————————————————— ###
    # top_k
    top_n_result1 =  df.groupby(['top_k'])['annual_return'].mean().rename('annual_return_mean')
    top_n_result2 =  df.groupby(['top_k'])['max_drawdown'].mean().rename('max_drawdown_mean')
    top_n_result3 =  df.groupby(['top_k'])['sharpe'].mean().rename('sharpe_mean')
    top_n_result4 =  df.groupby(['top_k'])['information_ratio_x'].mean().rename('information_ratio_mean')
    top_n_result5 = df.query('qlib_annual_excess_return > 0').groupby(['top_k'])['annual_return'].mean().rename('q_annual_return_mean')
    top_n_result9 = df.query('qlib_annual_excess_return > 0').groupby(['top_k'])['information_ratio_x'].mean().rename('q_information_ratio_mean')
    result1 = pd.concat([top_n_result1, top_n_result2, top_n_result3, top_n_result4, top_n_result5, top_n_result9], axis=1).reset_index()
    # print(result.to_string(max_cols=None))

    result1.to_csv(f'{save_path}/top_n_r.csv', index=False, date_format='%Y-%m-%d')

    # hold_p
    hold_p_result1 =  df.groupby(['hold_p'])['annual_return'].mean().rename('annual_return_mean')
    hold_p_result2 =  df.groupby(['hold_p'])['max_drawdown'].mean().rename('max_drawdown_mean')
    hold_p_result3 =  df.groupby(['hold_p'])['sharpe'].mean().rename('sharpe_mean')
    hold_p_result4 =  df.groupby(['hold_p'])['information_ratio_x'].mean().rename('information_ratio_mean')
    hold_p_result5 = df.query('qlib_annual_excess_return > 0').groupby(['hold_p'])['annual_return'].mean().rename('q_annual_return_mean')
    hold_p_result9 = df.query('qlib_annual_excess_return > 0').groupby(['hold_p'])['information_ratio_x'].mean().rename('q_information_ratio_mean')
    result2 = pd.concat([hold_p_result1, hold_p_result2, hold_p_result3, hold_p_result4, hold_p_result5, hold_p_result9], axis=1).reset_index()
    # print(result2.to_string(max_cols=None))


    result2.to_csv(f'{save_path}/hold_p_r.csv', index=False, date_format='%Y-%m-%d')
    
    # n_drop
    df_temp = df.copy()
    df_temp['ra'] = df_temp['n_drop'] / df_temp['top_k']
    df_temp['ra'] = np.where(df_temp['ra'] == 0.2, 0.3, np.where(df_temp['ra'] == 0.4, 0.5 , df_temp['ra']))
    n_drop_result1 = df_temp.groupby(['ra'])['annual_return'].mean().rename('annual_returnt_mean')
    n_drop_result2 = df_temp.query('qlib_annual_excess_return > 0').groupby(['ra'])['annual_return'].mean().rename('q_annual_return_mean')

    result3 = pd.concat([n_drop_result1, n_drop_result2], axis=1).reset_index()
    # print(result3.to_string(max_cols=None))

    result3.to_csv(f'{save_path}/n_drop_r.csv', index=False, date_format='%Y-%m-%d')

    ### top_k、n_drop、hold_p 整体分析 ———————————————————————————————————————————————————————————————————————————————————————— ###
    # 参数组合数量
    count_result = df.groupby(['top_k', 'n_drop', 'hold_p']).size().rename('count')
    
    # 参数组合超额收益平均值
    annual_return_mean = df.groupby(['top_k', 'n_drop', 'hold_p'])['excess_return'].mean().rename('annual_return_mean')

    # 参数组合超额收益区间分布数量
    excess_return_0_count = df.query('excess_return > 0').groupby(['top_k', 'n_drop', 'hold_p']).size().rename('excess_return_0_count')
    excess_return_0_50_count = df.query('excess_return > 0 and excess_return < 0.5').groupby(['top_k', 'n_drop', 'hold_p']).size().rename('excess_return_0_50_count')
    excess_return_50_100_count = df.query('excess_return >= 0.5 and excess_return < 1').groupby(['top_k', 'n_drop', 'hold_p']).size().rename('excess_return_50_100_count')
    excess_return_100_count = df.query('excess_return >= 1').groupby(['top_k', 'n_drop', 'hold_p']).size().rename('excess_return_100_count')

    # 参数组合超额收益平均值
    max_drawdown_mean = df.groupby(['top_k', 'n_drop', 'hold_p'])['max_drawdown'].mean().rename('max_drawdown_mean')

    # 合并数据
    result5 = pd.concat([count_result, 
                        annual_return_mean, excess_return_0_count, excess_return_0_50_count,excess_return_50_100_count,excess_return_100_count,
                        max_drawdown_mean
    ], axis=1).reset_index()

    result5 = result5.fillna(0)
    result5['excess_return_0_r'] = result5['excess_return_0_count'] / result5['count']
    result5['excess_return_0_50_r'] = result5['excess_return_0_50_count'] / result5['count']
    result5['excess_return_50_100_r'] = result5['excess_return_50_100_count'] / result5['count']
    result5['excess_return_100_r'] = result5['excess_return_100_count'] / result5['count']
    result5 = result5.drop(['excess_return_0_count', 'excess_return_0_50_count', 'excess_return_50_100_count', 'excess_return_100_count'], axis=1)
    # print(result5.to_string(max_cols=None))

    result5.to_csv(f'{save_path}/b_params_r.csv', index=False, date_format='%Y-%m-%d')

    ### b_params、t_params 整体分析 ——————————————————————————————————————————————————————————————————————————————————————————— ###

    df['b_params'] = df['top_k'].astype(str) + "-" + df['n_drop'].astype(str) + "-" + df['hold_p'].astype(str) 
    df['t_params'] = df['D_model'].astype(str) + "-" + df['LR'].astype(str) + "-" + df['Dropout'].astype(str) 

    e_count_result = df.groupby(['b_params', 't_params']).size().rename('count')
    e_annual_return_mean = df.groupby(['b_params', 't_params'])['excess_return'].mean().rename('annual_return_mean')

    e_excess_return_0_count = df.query('excess_return > 0').groupby(['b_params', 't_params']).size().rename('excess_return_0_count')
    e_excess_return_0_50_count = df.query('excess_return > 0 and excess_return < 0.5').groupby(['b_params', 't_params']).size().rename('excess_return_0_50_count')
    e_excess_return_50_100_count = df.query('excess_return >= 0.5 and excess_return < 1').groupby(['b_params', 't_params']).size().rename('excess_return_50_100_count')
    e_excess_return_100_count = df.query('excess_return >= 1').groupby(['b_params', 't_params']).size().rename('excess_return_100_count')

    e_max_drawdown_mean = df.groupby(['b_params', 't_params'])['max_drawdown'].mean().rename('max_drawdown_mean')

    result6 = pd.concat([
        e_count_result, e_annual_return_mean, e_excess_return_0_count, e_excess_return_0_50_count, e_excess_return_50_100_count, e_excess_return_100_count,e_max_drawdown_mean
    ], axis=1).reset_index()

    result6 = result6.fillna(0)
    result6['excess_return_0_r'] = result6['excess_return_0_count'] / result6['count']
    result6['excess_return_0_50_r'] = result6['excess_return_0_50_count'] / result6['count']
    result6['excess_return_50_100_r'] = result6['excess_return_50_100_count'] / result6['count']
    result6['excess_return_100_r'] = result6['excess_return_100_count'] / result6['count']
    result6 = result6.drop(['excess_return_0_count', 'excess_return_0_50_count', 'excess_return_50_100_count', 'excess_return_100_count'], axis=1)

    result6['b_params'] = "p" + result6['b_params']
    result6.to_csv(f'{save_path}/b_t_params_r.csv', index=False) # , date_format='%Y-%m-%d')
    # print(result6.to_string(max_cols=None))


    ### 种子筛选 —————————————————————————————————————————————————————————————————————————————————————————————————————————————— ###
    
    con_1 = df['qlib_annual_excess_return'] > 0                               # 超额收益大于0
    df_s = df[con_1]
    print(f"len(df_s):{len(df_s)}")
    
    con_2 = df['qlib_sharpe'] > df_s['qlib_sharpe'].mean()                      # sharpe大于中位值
    con_3 = df['information_ratio_y'] > df_s['information_ratio_y'].mean()      # 信息率大于中位值
    con_4 = df['Test_IC'] > df_s['Test_IC'].mean()                              # testic大于中位值
    con_5 = df['qlib_max_drawdown'] > df_s['qlib_max_drawdown'].mean()  

    con = con_1 & con_2 & con_3 & con_4 & con_5
    df_s = df[con]

    s_count_result = df_s.groupby(['top_k', 'n_drop', 'hold_p']).size().rename('count')
    s_annual_return_mean = df_s.groupby(['top_k', 'n_drop', 'hold_p'])['excess_return'].mean().rename('annual_return_mean')
    s_max_drawdown_mean = df_s.groupby(['top_k', 'n_drop', 'hold_p'])['max_drawdown'].mean().rename('max_drawdown_mean')

    # 合并数据
    result10 = pd.concat([s_count_result, s_annual_return_mean, s_max_drawdown_mean
    ], axis=1).reset_index()

    result5 = result5[['top_k', 'n_drop', 'hold_p', 'annual_return_mean', 'max_drawdown_mean']]
    result5 = result5.rename(columns={'annual_return_mean': 'annual_return_old', 'max_drawdown_mean': 'max_drawdown_old'})
    
    filter_results = pd.merge(result5,result10,on=['top_k', 'n_drop', 'hold_p'], how='right')

    filter_results['annual_return_c'] = filter_results['annual_return_mean'] - filter_results['annual_return_old'] 
    
    print(filter_results.to_string(max_cols=None))
    # print(result10.to_string(max_cols=None))

    # print("\n" + "*" * 100)
    # # print(f"len(df_s):{len(df_s)}")
    # print(f"len(df_s):{len(df_s)}")
    # print(f"qlib_sharpe:{df_s['qlib_sharpe'].mean()}")
    # print(f"information_ratio:{df_s['information_ratio_y'].mean()}")
    # print(f"test_ic:{df_s['Test_IC'].mean()}")
    


    
    #################################################################################################

    ### qlib_annual_excess_return >0 平均值 ——————————————————————————————————————————————————————————————————————————————————— ###
    ### qlib 回测结果筛选 ——————————————————————————————————————————————————————————— —————————————————————————————————————————— ###
    # q_annual_return_mean = df.query('qlib_annual_excess_return > 0 and qlib_sharpe >= 3 and information_ratio_y > 1.5 and Test_IC > 0.02 and Test_RIC > 0.02').groupby(['top_k', 'n_drop', 'hold_p'])['excess_return'].mean().rename('q_annual_return_mean')
    
    
    
   
    


def main(
    data_path: str = "/home/idc2/notebook/zxf/data/modoel_training/csi800_20251211_20150101_20251208/Backtest_Results",
    # "/home/idc2/notebook/zxf/data/modoel_training/csi800_20251211_20150101_20251208/Backtest_Results/qlib/backtest_qlib_results.csv"
    # data_path: str = "/home/idc2/notebook/zxf/data/modoel_training/csi800_20251211_20150101_20251208/Backtest_Results/Backtest_Results.csv"
):
    backtest_results_path = f"{data_path}/Backtest_Results.csv"
    
    df_results = read_results(backtest_results_path)
    results_analysis(df_results, data_path)
    
    # print(df_results)





if __name__ == "__main__":
    fire.Fire(main)