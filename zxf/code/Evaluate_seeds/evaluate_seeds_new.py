import numpy as np
import pandas as pd
import fire

def evaluate_seed_metrics(df_seed: pd.DataFrame, top_ratio=0.2):
    n = len(df_seed)
    top_k = max(1, int(n * top_ratio))

    mean_rankicir = df_seed['rankicir'].mean()
    std_rankicir = df_seed['rankicir'].std()
    cv_rankicir = std_rankicir / (abs(mean_rankicir) + 1e-6)

    mean_icir = df_seed['icir'].mean()
    mean_rankic = df_seed['rankic'].mean()
    mean_ic = df_seed['ic'].mean()

    top_rankicir_mean = (
        df_seed
        .sort_values('rankicir', ascending=False)
        .head(top_k)['rankicir']
        .mean()
    )

    mid = n // 2
    early_rankicir = df_seed.iloc[:mid]['rankicir'].mean()
    late_rankicir = df_seed.iloc[mid:]['rankicir'].mean()

    overfit_gap = early_rankicir - late_rankicir

    return {
        'mean_rankicir': mean_rankicir,
        'std_rankicir': std_rankicir,
        'cv_rankicir': cv_rankicir,
        'mean_icir': mean_icir,
        'mean_rankic': mean_rankic,
        'mean_ic': mean_ic,
        'top_rankicir_mean': top_rankicir_mean,
        'early_rankicir': early_rankicir,
        'late_rankicir': late_rankicir,
        'overfit_gap': overfit_gap
    }


def compute_seed_score(m):
    score = (
        0.45 * m['mean_rankicir']
        + 0.25 * m['mean_icir']
        + 0.15 * m['mean_rankic']
        + 0.10 * m['top_rankicir_mean']
        - 0.30 * m['cv_rankicir']
        - 0.50 * max(0, m['overfit_gap'])
    )
    return score


def classify_seed(m):
    """
    返回: (SeedType, RecommendedUsage)
    """

    # 方便书写
    mr = m['mean_rankicir']
    cv = m['cv_rankicir']
    top = m['top_rankicir_mean']
    overfit = m['overfit_gap']
    ric = m['mean_rankic']

    # 手术刀型
    if top > 0.7 and cv > 0.6:
        return '手术刀型', '仅用于 Ensemble（小权重）'

    # 老黄牛型
    if mr > 0.35 and cv < 0.35 and abs(overfit) < 0.05:
        return '老黄牛型', '主力模型 / 核心仓位'

    # 瑞士军刀型
    if mr > 0.45 and cv < 0.5:
        return '瑞士军刀型', '单模型或 Ensemble 核心'

    # 彩票型
    if top > 0.8 and mr < 0.2:
        return '彩票型', '研究用途，不回测'

    # 钝刀型
    if abs(mr) < 0.1 and cv < 0.4:
        return '钝刀型', '稳定器 / 风控辅助'

    # 纸糊刀型
    if overfit > 0.15:
        return '纸糊刀型', '淘汰'

    # 默认
    return '未分类', '人工复核'


def evaluate_and_classify_seeds(df_all: pd.DataFrame):
    results = []

    for seed, df_seed in df_all.groupby('Seed'):
        metrics = evaluate_seed_metrics(df_seed)
        score = compute_seed_score(metrics)
        seed_type, usage = classify_seed(metrics)

        results.append({
            'Seed': seed,
            'SeedScore': score,
            'SeedType': seed_type,
            'RecommendedUsage': usage,
            **metrics
        })

    result_df = pd.DataFrame(results)
    return result_df.sort_values('SeedScore', ascending=False)


def main(
    # seed_path: str = "/home/idc2/notebook/zxf/data/master_results/master_20251207_csi800_test_data/Backtest_Results/info_result.csv"
    data_path: str = "/home/idc2/notebook/zxf/data/master_results/csi1000_20251224_20150101_20251223/Backtest_Results",
    seed_filename: str = "info_result.csv",
):
    seed_path = f"{data_path}/{seed_filename}"


    df_seed = pd.read_csv(seed_path)

    df_valid = df_seed[['Seed', 'Step', 'Train_loss', 'Valid_IC','Valid_ICIR', 'Valid_RIC', 'Valid_RICIR']].rename(columns={
        'Valid_IC': 'ic','Valid_ICIR': 'icir', 'Valid_RIC': 'rankic', 'Valid_RICIR': 'rankicir'
    })

    df_test = df_seed[['Seed', 'Step', 'Train_loss', 'Test_IC','Test_ICIR', 'Test_RIC', 'Test_RICIR']].rename(columns={
        'Test_IC': 'ic','Test_ICIR': 'icir', 'Test_RIC': 'rankic', 'Test_RICIR': 'rankicir'
    })

    # print(df_valid)

    valid_results = evaluate_and_classify_seeds(df_valid)
    test_results = evaluate_and_classify_seeds(df_test)

    valid_results.to_csv(f'{data_path}/seed_valid_analysis_results.csv', index=False)
    test_results.to_csv(f'{data_path}/seed_test_analysis_results.csv', index=False)

    print("\n\n" + "-" * 100)
    print(valid_results.to_string(max_cols=None))
    print(test_results.to_string(max_cols=None))

    print(df_seed)


if __name__ == "__main__":
    fire.Fire(main)




"""



"""