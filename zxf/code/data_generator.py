import sys
import os
import yaml
import argparse
import fire
from datetime import datetime, date

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


def _parse_date(x):
    """
    解析日期参数，支持:
    - datetime/date
    - 'YYYY-MM-DD'
    - 'YYYYMMDD'
    """
    if x is None:
        return None
    if isinstance(x, (datetime, date)):
        return x
    s = str(x).strip()
    if s == "":
        return None
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"无法解析日期: {x}，请使用 YYYY-MM-DD 或 YYYYMMDD")


def _update_segments_and_times(
        config: dict,
        train_start=None, train_end=None,
        valid_start=None, valid_end=None,
        test_start=None, test_end=None,
):
    """将滚动日期写入 config 的各个相关位置，避免时间字段不一致导致的数据泄漏。"""
    ts = _parse_date(train_start)
    te = _parse_date(train_end)
    vs = _parse_date(valid_start)
    ve = _parse_date(valid_end)
    tes = _parse_date(test_start)
    tee = _parse_date(test_end)

    # 1) dataset segments
    seg = (
        config
        .get("task", {})
        .get("dataset", {})
        .get("kwargs", {})
        .get("segments", {})
    )
    if ts and te and "train" in seg:
        seg["train"] = [ts, te]
    if vs and ve and "valid" in seg:
        seg["valid"] = [vs, ve]
    if tes and tee and "test" in seg:
        seg["test"] = [tes, tee]

    # 2) handler / market_handler 的时间字段
    # 这些字段决定了特征处理器的 fit 范围；滚动时必须同步更新 fit_end_time=train_end
    def _patch_handler_times(h: dict):
        if not isinstance(h, dict):
            return
        if ts:
            h["start_time"] = ts
            h["fit_start_time"] = ts
        # end_time 一般需要覆盖到 test_end（因为要准备 test 特征）
        if tee:
            h["end_time"] = tee
        if te:
            h["fit_end_time"] = te

    _patch_handler_times(config.get("data_handler_config"))
    _patch_handler_times(config.get("market_data_handler_config"))

    # 3) port_analysis backtest 的时间字段（用于回测/记录）
    pa = config.get("port_analysis_config", {})
    bt = pa.get("backtest", {}) if isinstance(pa, dict) else {}
    if isinstance(bt, dict):
        if tes:
            bt["start_time"] = tes
        if tee:
            bt["end_time"] = tee

    # 4) 顶层日期字段（仅用于可读性/留档）
    # 注意：YAML anchor 在 safe_load 后已展开，这里只是为了把字段写回去，方便复现。
    if ts:
        config["train_date_start"] = ts
    if te:
        config["train_date_end"] = te
    if vs:
        config["valid_date_start"] = vs
    if ve:
        config["valid_date_end"] = ve
    if tes:
        config["test_date_start"] = tes
    if tee:
        config["test_date_end"] = tee

    return config

def data_generator(
        market_name: str="csi800b",
        qlib_path: str="/home/idc2/notebook/qlib_bin/cn_data_train",
        data_path: str="/home/idc2/notebook/zxf/data",
        folder_name: str="csi800_20251127_data",
        # 可选：优先读取/使用指定 config
        config_path: str = None,
        # 可选：滚动日期覆盖（增量训练/滚动切数据）
        train_start: str = None,
        train_end: str = None,
        valid_start: str = None,
        valid_end: str = None,
        test_start: str = None,
        test_end: str = None,
):

    qlib.init(provider_uri=qlib_path, region=REG_CN)
    save_path = f"{data_path}/master_results/{folder_name}"
    os.makedirs(save_path, exist_ok=True)

    # 读取 config：优先级
    # 1) 显式 config_path
    # 2) 当前实验目录下的 workflow_config（便于每次滚动有自己的日期段）
    # 3) data_path 下的默认 workflow_config
    default_config_path = f"{data_path}/workflow_config_master_Alpha158_{market_name}.yaml"
    local_config_path = f"{save_path}/workflow_config_master_Alpha158_{market_name}.yaml"
    cfg_path = config_path or (local_config_path if os.path.exists(local_config_path) else default_config_path)

    with open(cfg_path, 'r') as f:
        config = yaml.safe_load(f)

    # 如果传入了滚动日期覆盖，则同步更新 segments/handler times
    if any(x is not None and str(x).strip() != "" for x in [train_start, train_end, valid_start, valid_end, test_start, test_end]):
        config = _update_segments_and_times(
            config,
            train_start=train_start, train_end=train_end,
            valid_start=valid_start, valid_end=valid_end,
            test_start=test_start, test_end=test_end,
        )

    start_date = config["task"]["dataset"]["kwargs"]["segments"]["train"][0].strftime("%Y%m%d")
    end_date = config["task"]["dataset"]["kwargs"]["segments"]["test"][1].strftime("%Y%m%d")

    # 将本次实际使用的 config 落盘到实验目录，保证实验可复现（避免依赖外部默认 config）
    with open(local_config_path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)

    h_conf = config["task"]["dataset"]["kwargs"]["handler"]
    h_path = Path(f'{save_path}/' + f'handler_{start_date}_{end_date}.pkl')

    if not h_path.exists():
        h = init_instance_by_config(h_conf)
        h.to_pickle(h_path, dump_all=True)
        # print('Save preprocessed data to', h_path)

    config["task"]["dataset"]["kwargs"]["handler"] = f"file://{h_path}"
    dataset = init_instance_by_config(config['task']["dataset"])
    dl_train = dataset.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
    dl_valid = dataset.prepare("valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
    dl_test = dataset.prepare("test", col_set=["feature", "label"], data_key=DataHandlerLP.DK_I)

    with open(f'{save_path}/{market_name}_self_dl_train.pkl', 'wb') as file: pickle.dump(dl_train, file)
    with open(f'{save_path}/{market_name}_self_dl_valid.pkl', 'wb') as file: pickle.dump(dl_valid, file)
    with open(f'{save_path}/{market_name}_self_dl_test.pkl', 'wb') as file: pickle.dump(dl_test, file)


if __name__ == "__main__":
    fire.Fire(data_generator)