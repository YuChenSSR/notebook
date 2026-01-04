import pickle
import numpy as np
import pandas as pd
import fire

from scipy.stats import spearmanr
from sklearn.decomposition import PCA

### 读取数据
def read_pkl(data_dir):
    with open(f'{data_dir}/csi800_self_dl_train.pkl', 'rb') as f:
        dl_train = pickle.load(f)

    # print(dl_train.data.shape)
    return dl_train

def evaluate_features(df_data):

    ### 参数
    df = df_data.copy()
    # print(df.head(1).to_string(max_cols=None))

    # 列
    FACTORS = df.columns.difference(['datetime', 'instrument', '(Ref($adjclose,-5)/Ref($adjclose,-1)-1) / (1 + Std($adjclose/Ref($adjclose,1)-1, 5))'])
    DATE_COL = 'datetime'
    SYMBOL = 'instrument'
    LABEL_COL = '(Ref($adjclose,-5)/Ref($adjclose,-1)-1) / (1 + Std($adjclose/Ref($adjclose,1)-1, 5))'

    RANKIC_MEAN_TH = 0.02
    RANKIC_IR_TH = 0.3
    CORR_TH = 0.7
    LASSO_ALPHA = 0.001

    # 1. 因子健康检查（RobustZScore 后专用）
    def factor_health_check(df, factors):
        """
        针对已经做过 RobustZScoreNorm 的因子健康检查

        nan_ratio、zero_ratio
        <5% 好;  5%–15% 可接受; > 30% 差;
    
        std
        明显>0	正常
        接近0	常数因子
        极端大	可能含异常值
    
        [检查结果：zero 后续关注]
        """
        
        records = []   
        for f in factors:
            x = df[f]
    
            records.append({
                "factor": f,
                "nan_ratio": x.isna().mean(),
                "zero_ratio": (x == 0).mean(),
                "mean": x.mean(),
                "std": x.std(),
                "p01": x.quantile(0.01),
                "p99": x.quantile(0.99),
            })
        return pd.DataFrame(records).set_index("factor")

    # 2. 因子相关性 & 多重共线性
    def factor_corr_matrix(df, factors):
        """
        计算因子间相关性矩阵（Pearson）
        """
        return df[factors].corr()

    def drop_high_corr_factors(corr_df, threshold=0.85):
        """
        从相关性矩阵中找出需要剔除的高相关因子
        """
        upper = corr_df.where(
            np.triu(np.ones(corr_df.shape), k=1).astype(bool)
        )
        drop_factors = [
            col for col in upper.columns
            if (upper[col].abs() > threshold).any()
        ]
        return drop_factors
        
    # 3. RankIC 计算（横截面、按交易日）  
    def calc_daily_rankic(df, factor, label, date_col="datetime"):
        """
        按交易日计算 RankIC（Spearman）
        """
        ic_list = []
    
        for _, g in df.groupby(date_col):
            x = g[factor]
            y = g[label]
    
            valid = x.notna() & y.notna()
            if valid.sum() < 10:
                continue
    
            ic, _ = spearmanr(x[valid], y[valid])
            ic_list.append(ic)
        return pd.Series(ic_list)
        
    def factor_ic_summary(df, factors, label, date_col="datetime"):
        """
        汇总 IC / ICIR
        """
        records = []
    
        for f in factors:
            ic_series = calc_daily_rankic(df, f, label, date_col)
    
            records.append({
                "factor": f,
                "ic_mean": ic_series.mean(),
                "ic_std": ic_series.std(),
                "ic_ir": (
                    ic_series.mean() / ic_series.std()
                    if ic_series.std() > 0 else 0.0
                ),
                "ic_abs_mean": ic_series.abs().mean(),
                "ic_obs": len(ic_series)
            })
        return pd.DataFrame(records).set_index("factor")

    # 4. 因子综合筛选（核心逻辑）   
    def select_factors(
        health_df,
        ic_df,
        corr_df,
        nan_threshold=0.2,
        std_range=(0.3, 3.0),
        min_ic_abs=0.01,
        corr_threshold=0.85
    ):
        """
        综合 Health + IC + 相关性 选择最终因子
        """
    
        # ---------- Step 1: Health Check ----------
        candidates = health_df[
            (health_df["nan_ratio"] < nan_threshold) &
            (health_df["std"].between(*std_range))
        ].index.tolist()
    
        # ---------- Step 2: IC 过滤 ----------
        candidates = [
            f for f in candidates
            if abs(ic_df.loc[f, "ic_mean"]) >= min_ic_abs
        ]
    
        # ---------- Step 3: 多重共线性 ----------
        corr_sub = corr_df.loc[candidates, candidates]
        drop_factors = drop_high_corr_factors(corr_sub, corr_threshold)
    
        final_factors = [
            f for f in candidates if f not in drop_factors
        ]
    
        return final_factors
    
    # 5. PCA（可选，仅用于研究冗余）  
    def run_pca_analysis(df, factors, n_components=0.9):
        """
        PCA 仅用于研究因子冗余，不建议直接用于交易
        """
        X = df[factors].dropna()
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)
    
        explained = pca.explained_variance_ratio_.sum()
    
        return {
            "pca": pca,
            "explained_variance_sum": explained,
            "n_components": pca.n_components_,
            "X_pca": X_pca
        }
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
    # 6. 总控函数（你实际只需要调用这个）   
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
    
    def factor_analysis_pipeline(
        df,
        factors,
        label="label",
        date_col="datetime"
    ):
        """
        一键完成完整因子分析流程
        """
        # 1. Health Check
        health = factor_health_check(df, factors)
        # print(health)

        # 2. IC 分析
        ic_summary = factor_ic_summary(df, factors, label, date_col)
        # print(ic_summary)
    
        # 3. 相关性分析
        corr = factor_corr_matrix(df, factors)
        # print(corr)
    
        # 4. 综合筛选
        selected_factors = select_factors(
            health_df=health,
            ic_df=ic_summary,
            corr_df=corr
        )
    
        return {
            "health": health,
            "ic_summary": ic_summary,
            "corr": corr,
            "selected_factors": selected_factors
        }
    

    result = factor_analysis_pipeline(
        df=df,
        factors=FACTORS,
        label=LABEL_COL,
        date_col=DATE_COL,
    )
    
    health_df = result["health"]
    ic_df = result["ic_summary"]
    corr_df = result["corr"]
    final_factors = result["selected_factors"]

    print("最终可用因子：", final_factors)
        
    

def main(
    data_dir: str = f"/home/idc2/notebook/zxf/data/master_results/csi800_20251229_20150101_20251226"
):
    pkl_data = read_pkl(data_dir)
    df = pkl_data.data
    df = df.droplevel(0, axis=1)
    df.reset_index(inplace=True)



    evaluate_features(df)
    



if __name__ == "__main__":
    fire.Fire(main)



