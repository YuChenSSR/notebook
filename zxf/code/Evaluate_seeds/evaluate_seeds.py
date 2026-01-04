import pandas as pd
import numpy as np

def evaluate_seed(df_seed: pd.DataFrame,
                  top_ratio=0.2,
                  w_rankicir=0.4,
                  w_icir=0.3,
                  w_rankic=0.2,
                  w_stability=0.1):
    """
    对单个 seed 的训练日志进行量化评估
    """

    n = len(df_seed)
    top_k = max(1, int(n * top_ratio))

    # ========= 基础统计 =========
    mean_rankicir = df_seed['rankicir'].mean()
    mean_icir = df_seed['icir'].mean()
    mean_rankic = df_seed['rankic'].mean()

    std_rankicir = df_seed['rankicir'].std()
    cv_rankicir = std_rankicir / (abs(mean_rankicir) + 1e-6)

    # ========= Top-k 表现 =========
    top_rankicir_mean = (
        df_seed
        .sort_values('rankicir', ascending=False)
        .head(top_k)['rankicir']
        .mean()
    )

    # ========= 时间结构（过拟合惩罚） =========
    mid = n // 2
    early_mean = df_seed.iloc[:mid]['rankicir'].mean()
    late_mean = df_seed.iloc[mid:]['rankicir'].mean()

    overfit_penalty = max(0, early_mean - late_mean)

    # ========= 综合评分 =========
    score = (
        w_rankicir * mean_rankicir
        + w_icir * mean_icir
        + w_rankic * mean_rankic
        + w_stability * top_rankicir_mean
        - 0.3 * cv_rankicir
        - 0.5 * overfit_penalty
    )

    return {
        'mean_rankicir': mean_rankicir,
        'mean_icir': mean_icir,
        'mean_rankic': mean_rankic,
        'std_rankicir': std_rankicir,
        'cv_rankicir': cv_rankicir,
        'top_rankicir_mean': top_rankicir_mean,
        'early_rankicir': early_mean,
        'late_rankicir': late_mean,
        'overfit_penalty': overfit_penalty,
        'seed_score': score,
    }

def evaluate_all_seeds(df_all: pd.DataFrame):
    results = []

    for seed, df_seed in df_all.groupby('Seed'):
        metrics = evaluate_seed(df_seed)
        metrics['Seed'] = seed
        results.append(metrics)

    result_df = pd.DataFrame(results)

    return result_df.sort_values('seed_score', ascending=False)
