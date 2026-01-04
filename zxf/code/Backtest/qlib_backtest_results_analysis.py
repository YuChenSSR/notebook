import pandas as pd
import fire

def read_csv(data_path):
    return pd.read_csv(data_path)


def pred_analysis(df_backtest_results, top_n):

    df = df_backtest_results.copy()

    ### 过滤条件
    con_1 = df['annual_excess_return'] > 0
    con_2 = df['sharpe'] > 0
    con_3 = df['information_ratio'] > 0
    con = con_1 & con_2 & con_3
    df = df[con]



    ### 排名&权重
    df_rank = df.copy()
    
    df_rank = df_rank.sort_values(by="annual_return")
    if len(df_rank) >= top_n:
        df_rank = df_rank[-top_n:]
    
    for col in df_rank.columns:
        if col in ["Seed", "Step", "Train_loss"]:
            continue
        df_rank[f'{col}_rank'] = df_rank[col].rank(method='dense', ascending=True)

    weights = {
        'Valid_IC_rank': 0.005,
        'Valid_ICIR_rank': 0.005,
        'Valid_RIC_rank': 0.005,
        'Valid_RICIR_rank': 0.005,
        'Test_IC_rank': 0.015,
        'Test_ICIR_rank': 0.05,
        'Test_RIC_rank': 0.015,
        'Test_RICIR_rank': 0.05,
        'annual_return_rank': 0.45,
        'max_drawdown_rank':  0.1,
        'sharpe_rank': 0.15,
        'information_ratio_rank': 0.15, 
    }
    df_rank['rank_score'] = df_rank[list(weights.keys())].mul(pd.Series(weights)).sum(axis=1)
    # df_rank = df_rank.sort_values(by='score')
    # print(df_rank)



    ### 中位值信息
    df_median = df.groupby('Seed').agg({
        'annual_return': 'median',
        'sharpe': 'median',
        'information_ratio': 'median'
    }).reset_index()
    df_median = df_median.rename(columns={'annual_return': 'annual_return_median','sharpe': 'sharpe_median','information_ratio': 'information_ratio_median'})
    
    # print(df_median)

    ### 前30中占比
    df_percent = df.copy()
    df_percent = df_percent.sort_values(by='annual_return')

    if len(df_percent) >= top_n:
        df_percent = df_percent[-top_n:]

    df_seed_percent = df_percent.groupby('Seed').size().reset_index(name='seed_percent')
    df_seed_percent['seed_percent'] = df_seed_percent['seed_percent'] / top_n



    ### 汇总结果
    df_reuslts = pd.merge(df_rank, df_median, on="Seed", how='left')
    df_reuslts = pd.merge(df_reuslts, df_seed_percent, on="Seed", how='left')



    rank_score_min = df_reuslts['rank_score'].min()
    rank_score_max = df_reuslts['rank_score'].max()
    df_reuslts['rank_score'] = (df_reuslts['rank_score'] - rank_score_min) / (rank_score_max-rank_score_min) * 100

    seed_percent_min = df_reuslts['seed_percent'].min()
    seed_percent_max = df_reuslts['seed_percent'].max()
    df_reuslts['seed_percent'] = (df_reuslts['seed_percent'] - seed_percent_min) / (seed_percent_max-seed_percent_min) * 100

    df_reuslts['median_score'] = df_reuslts['annual_return_median'] + df_reuslts['sharpe_median'] + df_reuslts['information_ratio_median']

    # print(df_reuslts)
    
    median_score_min = df_reuslts['median_score'].min()
    median_score_max = df_reuslts['median_score'].max()
    df_reuslts['median_score'] = (df_reuslts['median_score'] - median_score_min) / (median_score_max-median_score_min) * 100
    
    
    df_reuslts['score'] = df_reuslts['rank_score'] * 0.6 + df_reuslts['seed_percent'] * 0.2 + df_reuslts['median_score'] * 0.2


    df_reuslts = df_reuslts.sort_values(by='score')
    df_reuslts = df_reuslts[['Seed', 'Step', 'Valid_IC', 'Valid_ICIR', 'Valid_RIC', 'Valid_RICIR', 'Test_IC', 'Test_ICIR', 'Test_RIC', 'Test_RICIR', 'annual_return', 'max_drawdown', 'sharpe', 'information_ratio', 'seed_percent', 'median_score',  'score']]
    
    print(df_reuslts.to_string(max_cols=None))





def main(
    top_n = 30,
    backtest_results_file = f"/home/idc2/notebook/zxf/data/master_results/master_20251205_csi800_test_data/Backtest_Results/info_result.csv"
):
    # get backtest results
    backtest_results = read_csv(backtest_results_file)

    # results analysis
    pred_analysis(backtest_results, top_n)


if __name__ == "__main__":
    fire.Fire(main)

