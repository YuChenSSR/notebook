import pickle
import fire
import numpy as np
import pandas as pd
from loguru import logger


def read_data(data_dir):
    with open(data_dir, 'rb') as f:
        dl_train = pickle.load(f)
    return dl_train

def main(
    # data_dir: str = "/home/a/notebook/zxf/data/Daily_data/Beta_data/model_232_128",
    # data_dir: str = "/home/a/notebook/zxf/data/Daily_data/Training_data/csi8",
    data_dir: str = "/home/idc2/notebook/quant/data/experimental_results/csi800_771_20260121_20150101_20260116",


    # notebook/zxf/data/Daily_data/Good_seed/seed7/csi800_self_dl_test_raw.pkl

):

    # csi800_self_dl_test.pkl
    
    data_path = f"{data_dir}/csi800_self_dl_valid.pkl"
    
    ### 读取数据
    dl_train = read_data(data_path)
    print(dl_train.data)
    for col in dl_train.data.columns:
        print(col)
    print(len(dl_train.data.columns))


    # df = dl_train.data
    # df = df.droplevel(0, axis=1)
    # df.reset_index(inplace=True)

    # for col in df.columns:
    #     col = "'" + col + "',"
    #     print(col)
        
    # df = df[df['instrument'] == 'SH600000']

    # df = df[(df['datetime'] >= "2025-05-01") & (df['datetime'] <= "2025-05-30")]

    # df = df[["datetime", "instrument", "Ref($adjclose, -5) / Ref($adjclose, -1) - 1"]]
    # print(df)

   
    
if __name__ == "__main__":
    fire.Fire(main)





