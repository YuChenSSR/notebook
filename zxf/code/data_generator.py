import sys
import os
import yaml
import argparse
import fire
from datetime import date

import pprint as pp
import numpy as np
import pickle
from pathlib import Path



# 将本地qlib目录设置为最高优先级
QLIB_DIRNAME = '/home/idc2/notebook/qlib'
sys.path.insert(0, QLIB_DIRNAME)


import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord
from qlib.tests.data import GetData
from qlib.data.dataset.handler import DataHandlerLP

def data_generator(
        market_name: str="csi800",
        qlib_path: str="/home/idc2/notebook/qlib_bin/cn_data_train",
        data_path: str="/home/idc2/notebook/zxf/data",
        folder_name: str="csi800_20260106_20150101_20260106",
):

    qlib.init(provider_uri=qlib_path, region=REG_CN)
    with open(f"{data_path}/workflow_config_master_Alpha158_{market_name}.yaml", 'r') as f:
        config = yaml.safe_load(f)

    start_date = config["task"]["dataset"]["kwargs"]["segments"]["train"][0].strftime("%Y%m%d")
    end_date = config["task"]["dataset"]["kwargs"]["segments"]["test"][1].strftime("%Y%m%d")


    save_path = f"{data_path}/master_results/{folder_name}"
    os.makedirs(save_path, exist_ok=True)

    h_conf = config["task"]["dataset"]["kwargs"]["handler"]
    h_path = Path(f'{save_path}/' + f'handler_{start_date}_{end_date}.pkl')

    if not h_path.exists():
        h = init_instance_by_config(h_conf)
        h.to_pickle(h_path, dump_all=True)
        del h

    config["task"]["dataset"]["kwargs"]["handler"] = f"file://{h_path}"
    dataset = init_instance_by_config(config['task']["dataset"])
    dl_train = dataset.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
    with open(f'{save_path}/{market_name}_self_dl_train.pkl', 'wb') as file: pickle.dump(dl_train, file)
    del dl_train
    
    dl_valid = dataset.prepare("valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
    with open(f'{save_path}/{market_name}_self_dl_valid.pkl', 'wb') as file: pickle.dump(dl_valid, file)
    del dl_valid
    
    dl_test = dataset.prepare("test", col_set=["feature", "label"], data_key=DataHandlerLP.DK_I)
    with open(f'{save_path}/{market_name}_self_dl_test.pkl', 'wb') as file: pickle.dump(dl_test, file)
    del dl_test

if __name__ == "__main__":
    fire.Fire(data_generator)