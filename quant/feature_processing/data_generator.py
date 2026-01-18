import sys
from pathlib import Path
DIRNAME = Path(__file__).absolute().resolve().parent
sys.path.append(str(DIRNAME))
sys.path.append(str(DIRNAME.parent.parent.parent))

# xue: 将本地qlib目录设置为最高优先级
QLIB_DIRNAME = '/home/idc2/notebook/qlib'
sys.path.insert(0, QLIB_DIRNAME)

import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord
from qlib.tests.data import GetData
from qlib.data.dataset.handler import DataHandlerLP

import os
import datetime
import shutil
import yaml
import argparse
import fire
from typing import Union, Optional


import pprint as pp
import numpy as np
import pandas as pd
import pickle
from pathlib import Path


def data_generator(
        market_name: str="csi800",
        data_path: str="/home/idc2/notebook/zxf/quant/data",
        qlib_data_path: str="/home/idc2/notebook/qlib_bin/cn_data_train",
        qlib_contrib_path: str="/home/idc2/notebook/qlib/qlib/contrib/data",
        today_date: Union[str, None] = None,
):
    save_path = data_path
    os.makedirs(save_path, exist_ok=True)

    if today_date is None:
        new_date = datetime.date.today()
    else:
        new_date = datetime.datetime.strptime(today_date, "%Y-%m-%d").date()


    # 备份原handler文件
    # h_old_file = Path("/home/a/notebook/qlib/qlib/contrib/data/handler.py")
    # h_new_file = Path("/home/a/notebook/qlib/qlib/contrib/data/handler_bak_bak_bak.py")
    h_old_file = Path(f"{qlib_contrib_path}/handler.py")
    h_new_file = Path(f"{qlib_contrib_path}/handler_bak_bak_bak.py")

    if h_old_file.exists():
        h_old_file.rename(h_new_file)

    # 备份原dataset文件
    d_old_file = Path(f"{qlib_contrib_path}/dataset.py")
    d_new_file = Path(f"{qlib_contrib_path}/dataset_bak_bak_bak.py")
    if d_old_file.exists():
        d_old_file.rename(d_new_file)
    
    # 复制my handler
    h_src = Path(f"{save_path}/handler.py")
    h_dst = Path(f"{qlib_contrib_path}/handler.py")
    shutil.copy2(h_src, h_dst)

    # 复制my dataset
    d_src = Path(f"{save_path}/dataset.py")
    d_dst = Path(f"{qlib_contrib_path}/dataset.py")
    shutil.copy2(d_src, d_dst)

    qlib.init(provider_uri=qlib_data_path, region=REG_CN)
    config_path = f"{data_path}/workflow_config_master_Alpha158_{market_name}.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # 修正日期
    config["data_handler_config"]["end_time"] = new_date
    config["task"]["dataset"]["kwargs"]["segments"]["test"][1] = new_date
    config["market_data_handler_config"]["end_time"] = new_date

    start_date = config["task"]["dataset"]["kwargs"]["segments"]["train"][0].strftime("%Y%m%d")
    end_date = config["task"]["dataset"]["kwargs"]["segments"]["test"][1].strftime("%Y%m%d")
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
    
    # 删除过程文件
    os.remove(h_path)

    # 删除my handler, dataset
    os.remove(h_dst)
    os.remove(d_dst)

    # 恢复原handler, dataset
    if h_new_file.exists():
        h_new_file.rename(h_old_file)
    if d_new_file.exists():
        d_new_file.rename(d_old_file)


if __name__ == "__main__":
    fire.Fire(data_generator)