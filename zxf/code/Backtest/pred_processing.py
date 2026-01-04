import pandas as pd
import fire


def read_csv(data_path):
    return pd.read_csv(data_path)



def main(
    data_path: str = f"/home/idc2/notebook/zxf/data/beta_data/",
    expt_no: str = f"csi800c_20251201_20150101_20251128",

):

    pred_path = f"{data_path}/{expt_no}/pred_50_lin.csv"
    df = read_csv(pred_path)

    ### chen
    # 原始a
    # df_a = df[['datetime', 'instrument', 'expected_rate']]
    # df_a = df_a.rename(columns={'expected_rate': 'score'})
    # df_a = df_a[~df_a['score'].isna()]
    # df_a.to_csv(f"{data_path}/{expt_no}/pred_50_chen_a.csv", index=False, date_format='%Y-%m-%d')

    # # 组合b
    # con_1 = df['expected_rate'] > 0
    # con_2 = df['expected_max_rate'] > 0
    # conditions = con_1 | con_2
    # df_b = df[conditions]
    # df_b = df_b[['datetime', 'instrument', 'score']]
    # df_b.to_csv(f"{data_path}/{expt_no}/pred_50_chen_b.csv", index=False, date_format='%Y-%m-%d')

    ### lin
    # 原始a
    df_lin_a = df[['datetime', 'instrument', 'pred_mixer']]
    df_lin_a = df_lin_a.rename(columns={'pred_mixer': 'score'})
    df_lin_a = df_lin_a[~df_lin_a['score'].isna()]
    df_lin_a.to_csv(f"{data_path}/{expt_no}/pred_50_lin_a.csv", index=False, date_format='%Y-%m-%d')
    print(df_lin_a)

    # 组合b
    df_lin_b = df[df['pred_mixer'] > 0]
    df_lin_b = df_lin_b[['datetime', 'instrument', 'score']]
    df_lin_b.to_csv(f"{data_path}/{expt_no}/pred_50_lin_b.csv", index=False, date_format='%Y-%m-%d')
    
    print(df_lin_b)

if __name__ == "__main__":
    fire.Fire(main)
