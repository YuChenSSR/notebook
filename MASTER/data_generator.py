#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.
"""
Qlib provides two kinds of interfaces. 
(1) Users could define the Quant research workflow by a simple configuration.
(2) Qlib is designed in a modularized way and supports creating research workflow by code just like building blocks.

The interface of (1) is `qrun XXX.yaml`.  The interface of (2) is script like this, which nearly does the same thing as `qrun XXX.yaml`
"""
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

import yaml
import argparse
import os
import pprint as pp
import numpy as np
import pickle


def parse_args():
    """parse arguments. You can add other arguments if needed."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--only_backtest", action="store_true", help="whether only backtest or not")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    # use default data
    # provider_uri = "~/.qlib/qlib_data/cn_data"  # target_dir
    # provider_uri = "~/notebook/qlib_bin"  # target_dir
    provider_uri = "~/notebook/qlib_bin/cn_data_train"  # target_dir
    # GetData().qlib_data(target_dir=provider_uri, region=REG_CN, exists_skip=True)
    qlib.init(provider_uri=provider_uri, region=REG_CN)
    with open("./workflow_config_master_Alpha158.yaml", 'r') as f:
        config = yaml.safe_load(f)

    h_conf = config["task"]["dataset"]["kwargs"]["handler"]
    h_path = DIRNAME / f'handler_{config["task"]["dataset"]["kwargs"]["segments"]["train"][0].strftime("%Y%m%d")}' \
                       f'_{config["task"]["dataset"]["kwargs"]["segments"]["test"][1].strftime("%Y%m%d")}.pkl'
    # if not h_path.exists():
    h = init_instance_by_config(h_conf)
    h.to_pickle(h_path, dump_all=True)
    print('Save preprocessed data to', h_path)
    
    config["task"]["dataset"]["kwargs"]["handler"] = f"file://{h_path}"
    dataset = init_instance_by_config(config['task']["dataset"])
    dl_train = dataset.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
    dl_valid = dataset.prepare("valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
    dl_test = dataset.prepare("test", col_set=["feature", "label"], data_key=DataHandlerLP.DK_I)

    universe = config["market"] # ['csi300','csi800']
    prefix = 'opensource' # ['original','opensource'], which training data are you using
    train_data_dir = f'data/self_exp'
    predict_data_dir = f'data/self_exp/opensource'

    with open(f'{train_data_dir}/{prefix}/{universe}_self_dl_train.pkl', 'wb') as file: pickle.dump(dl_train, file)
    with open(f'{predict_data_dir}/{universe}_self_dl_valid.pkl', 'wb') as file: pickle.dump(dl_valid, file)
    with open(f'{predict_data_dir}/{universe}_self_dl_test.pkl', 'wb') as file: pickle.dump(dl_test, file)
