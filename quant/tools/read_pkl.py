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
    data_dir: str = "/home/a/notebook/zxf/data/master_results/csi800_128_20260117_20250101_20260116",


    # notebook/zxf/data/Daily_data/Good_seed/seed7/csi800_self_dl_test_raw.pkl

):

    # csi800_self_dl_test.pkl
    
    data_path = f"{data_dir}/csi800_self_dl_test.pkl"
    
    ### 读取数据
    dl_train = read_data(data_path)
    print(dl_train.data)
    for col in dl_train.data.columns:
        print(col)
    # print(len(dl_train.data.columns))
   
    
if __name__ == "__main__":
    fire.Fire(main)





