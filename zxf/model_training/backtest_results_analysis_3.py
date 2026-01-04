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
    df['excess_return'] = df['annual_return'] - df['annual_benchmark_return']
    
    df = df[
        (df['excess_return'] > 0.2) &                                     # 必须有超额收益
        (df['volatility'] < df['volatility'].median()) & 
        (df['winning_rate_cumulant'] > max(0.5, df['winning_rate_cumulant'].median())) & 
        (df['sharpe'] > df['sharpe'].median()) &                          # 风险调整收益为正
    
        (df['Test_IC'] > df['Test_IC'].median()) &
        (df['Test_ICIR'] > df['Test_ICIR'].median()) &

        (df['Test_RIC'] > df['Test_RIC'].median()) &
        (df['Test_RICIR'] > max(0.3, df['Test_RICIR'].median())) &

    
        (df['annual_turnover'] < df['annual_turnover'].median()) &        # 防止极端换手
        (df['max_drawdown'] > df['max_drawdown'].median())                # 防止崩溃型模型
    ].reset_index(drop=True)

    return df
    # print(len(df))
    # print(df.tail(1).to_string(max_cols=None))


def pareto_front(df, maximize_cols, minimize_cols):

    data = df.reset_index(drop=True)
    n = len(data)
    is_pareto = np.ones(n, dtype=bool)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            better_or_equal = True
            strictly_better = False

            # maximize 指标
            for col in maximize_cols:
                if data.loc[j, col] < data.loc[i, col]:
                    better_or_equal = False
                    break
                if data.loc[j, col] > data.loc[i, col]:
                    strictly_better = True

            # minimize 指标
            if better_or_equal:
                for col in minimize_cols:
                    if data.loc[j, col] > data.loc[i, col]:
                        better_or_equal = False
                        break
                    if data.loc[j, col] < data.loc[i, col]:
                        strictly_better = True

            if better_or_equal and strictly_better:
                is_pareto[i] = False
                break

    return data[is_pareto]


def final_rank_score(df):
    df = df.copy()

    df['final_score'] = (
        df['max_drawdown'].rank(pct=True) * 0.2 +
        df['annual_return'].rank(pct=True) * 0.1 +
        df['winning_rate_cumulant'].rank(pct=True) * 0.15 +
        df['Test_IC'].rank(pct=True) * 0.1 +
        df['Test_ICIR'].rank(pct=True) * 0.05 +
        df['sharpe'].rank(pct=True) * 0.1 +
        df['information_ratio'].rank(pct=True) * 0.05 -
        df['annual_turnover'].rank(pct=True) * 0.05 -
        df['volatility'].rank(pct=True) * 0.15
    )

    return df.sort_values('final_score', ascending=False)





def main(
    data_path: str = "/home/idc2/notebook/zxf/data/modoel_training/csi800_20251211_20150101_20251208/Backtest_Results",
):    
    backtest_results_path = f"{data_path}/Backtest_Results.csv"
    df_results = read_results(backtest_results_path)
    df_results = df_results.rename(columns={'information_ratio_x': 'information_ratio', 'information_ratio_y': 'qlib_information_ratio'})

    backtest_results_path1 = f"{data_path}/Backtest_Results_new1.csv"
    df_results1 = read_results(backtest_results_path1)


    df_results = pd.concat([df_results, df_results1], ignore_index=True)
    # print(df_results.columns)
    # print(df_results)
    
    
    
    df_work = results_analysis(df_results)

    maximize_cols = [
        'annual_return',
        'max_drawdown',          # 负数，越大越好
        'sharpe',
        'information_ratio',
        'alpha',
        'winning_rate_cumulant',
        'Test_IC',
        'Test_ICIR',
        'Test_RIC', 
        'Test_RICIR'
    ]

    minimize_cols = [
        'annual_turnover',
        'volatility'
    ]
    
    pareto_df = pareto_front(
        df_work,
        maximize_cols=maximize_cols,
        minimize_cols=minimize_cols
    )


    seed_cols = [
        'Seed', 'D_model', 'LR', 'Dropout', 'Step',
        'top_k', 'n_drop', 'hold_p'
    ]
    
    metric_cols = maximize_cols + minimize_cols
    
    pareto_result = pareto_df[seed_cols + metric_cols] \
        .reset_index(drop=True)

    final_rank_df = final_rank_score(pareto_result)

    print(final_rank_df.to_string(max_cols=None))

if __name__ == "__main__":
    fire.Fire(main)