import pandas as pd
from loguru import logger
import pickle
import yaml
import torch
import fire
import sys

# from Master.master import MASTERModel

def main(
        market_name: str = "csi800b",
        folder_name: str = "csi800b_20251127_data",
        data_path: str = "/home/idc2/notebook/zxf/data/master_results",
        # seed: int = 4,
        # step: int = 22,
):

    
    ### 1.读取数据
    experiment_dir = f'{data_path}/{folder_name}'

    with open(f'{experiment_dir}/{market_name}_self_dl_train.pkl', 'rb') as f:
        dl_train = pickle.load(f)
    with open(f'{experiment_dir}/{market_name}_self_dl_valid.pkl', 'rb') as f:
        dl_valid = pickle.load(f)
    with open(f'{experiment_dir}/{market_name}_self_dl_test.pkl', 'rb') as f:
        dl_test = pickle.load(f)
    logger.info(f"Data loaded: {experiment_dir}")

   

    ### 2. 格式转换
    # train
    df_train = dl_train.data
    df_train.columns = df_train.columns.get_level_values(-1)
    df_train = df_train.reset_index()

    # test
    df_valid = dl_valid.data
    df_valid.columns = df_valid.columns.get_level_values(-1)
    df_valid = df_valid.reset_index()

    # test
    df_test = dl_test.data
    df_test.columns = df_test.columns.get_level_values(-1)
    df_test = df_test.reset_index()


    ### 3. 合并
    df_data = pd.concat([df_train, df_valid, df_test], ignore_index=True)
    df_data = df_data.drop_duplicates(subset=["datetime", "instrument"], keep="first")


    ### 4. 过滤
    instrument = df_data[['instrument']].drop_duplicates()
    first_date = df_data['datetime'].min()

    df_results = pd.DataFrame()
    i=0
    for _, row in instrument.iterrows():
        code = row['instrument']
        df_temp = df_data[df_data['instrument']==code]
        temp_first_date = df_temp['datetime'].min()
        if temp_first_date<=first_date:        
            df_results = pd.concat([df_results, df_temp], ignore_index=True)
            i+=1
            print(f"{code}, concat")
        else:
            print(f"{code}, exclude")


    # concat
    out_filename =f"{data_path}/{folder_name}/{market_name}_merge_exp_data.parquet"
    df_results = df_results.sort_values(by=['datetime','instrument'])
    df_results.to_parquet(out_filename)

    logger.success(f"Data Merge Completed...Count:{i}")

if __name__ == "__main__":
    fire.Fire(main)