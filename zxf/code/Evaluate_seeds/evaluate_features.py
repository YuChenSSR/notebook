import pickle
import numpy as np
import pandas as pd
import fire

from scipy.stats import spearmanr
from sklearn.decomposition import PCA

### 读取数据
def read_pkl(data_dir):
    with open(f'{data_dir}/csi800_self_dl_valid.pkl', 'rb') as f:
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

    # L1：截面标准化 & 数据健康
    def cross_sectional_zscore(df, factors):
        df = df.copy()
        df[factors] = (
            df.groupby(DATE_COL)[factors]
              .transform(lambda x: (x - x.mean()) / x.std())
        )
        # print(df.head(1).to_string(max_cols=None))
        return df
    def factor_health_check(df, factors):
        health = pd.DataFrame({
            'nan_ratio': df[factors].isna().mean(),
            'zero_ratio': (df[factors] == 0).mean(),
            'std': df[factors].std()
        })
        return health

    # L2：单因子截面 RankIC
    def calc_daily_rankic(df, factor):
        return (
            df.groupby(DATE_COL)
              .apply(lambda x: spearmanr(x[factor], x[LABEL_COL])[0])
        )
    
    
    def calc_rankic_stats(df, factors):
        rankic_ts = {}
        for f in factors:
            rankic_ts[f] = calc_daily_rankic(df, f)
    
        rankic_ts = pd.DataFrame(rankic_ts)
    
        stat = pd.DataFrame({
            'rankic_mean': rankic_ts.mean(),
            'rankic_std': rankic_ts.std(),
        })
        stat['rankic_ir'] = stat['rankic_mean'] / stat['rankic_std']
        stat['sign_consistency'] = (rankic_ts > 0).mean()
    
        return rankic_ts, stat

    # L3：时间稳定性（滚动 IR）
    def rolling_rankic_ir(rankic_ts, window=60):
        return rankic_ts.rolling(window).mean() / rankic_ts.rolling(window).std()


    # L4：截面冗余（相关性 + PCA）
    def mean_cross_sectional_corr(df, factors):
        corr = (
            df.groupby(DATE_COL)[factors]
              .corr()
              .groupby(level=1)
              .mean()
        )
        return corr
    
    
    def pca_diagnosis(df, factors):
        X = df[factors].dropna()
        X = StandardScaler().fit_transform(X)
        pca = PCA().fit(X)
        return pca.explained_variance_ratio_.cumsum()    


    # L5：联合建模（Lasso）
    def lasso_selection(df, factors):
        df = df.dropna(subset=factors + [LABEL_COL])
        X = df[factors]
        y = df[LABEL_COL]
    
        model = Lasso(alpha=LASSO_ALPHA)
        model.fit(X, y)
    
        coef = pd.Series(model.coef_, index=factors)
        return coef

    # ======================


    # 数据健康
    health = factor_health_check(df, FACTORS)
    """
    nan_ratio、zero_ratio
    <5% 好;  5%–15% 可接受; > 30% 差;

    std
    明显>0	正常
    接近0	常数因子
    极端大	可能含异常值

    [检查结果：zero 后续关注]
    """
    # df_nan = health[health['nan_ratio'] > 0.05]
    # df_zero = health[health['zero_ratio'] > 0.05]
    # df_std = health[health['std'] < 0.5]   
    # print(df_std)
    # print(len(df_std)/len(health))

    # L1：标准化
    # df = cross_sectional_zscore(df, FACTORS)
    

def main(
    data_dir: str = f"/home/idc2/notebook/zxf/data/master_results/csi800_20251229_20150101_20251226"
):
    pkl_data = read_pkl(data_dir)
    df = pkl_data.data
    df = df.droplevel(0, axis=1)
    df.reset_index(inplace=True)

    # df = df[df['instrument'] == 'SH600588']
    # df = df[(df['datetime'] >= '2025-06-01') & (df['datetime'] <= '2025-06-30')]

    evaluate_features(df)
    



if __name__ == "__main__":
    fire.Fire(main)



