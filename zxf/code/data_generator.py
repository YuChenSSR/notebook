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
import time



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
    # 同步更新 task.dataset.kwargs 内部的 handler/market_data_handler_config（避免 YAML anchor 未共享引用时失效）
    try:
        handler_kwargs = (
            config.get("task", {})
            .get("dataset", {})
            .get("kwargs", {})
            .get("handler", {})
            .get("kwargs", None)
        )
        _patch_handler_times(handler_kwargs)
    except Exception:
        pass
    try:
        mkt_kwargs = (
            config.get("task", {})
            .get("dataset", {})
            .get("kwargs", {})
            .get("market_data_handler_config", None)
        )
        _patch_handler_times(mkt_kwargs)
    except Exception:
        pass

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
        # handler 落盘模式：
        # - False（推荐，默认）：轻量保存，仅保存 handler 配置与处理器，不保存大体量数据（避免 OOM）
        # - True：完整保存（会把 _data/_infer/_learn 等私有大对象也序列化，通常非常大，容易被系统杀死）
        handler_dump_all: bool = False,
        # 可选：滚动日期覆盖（增量训练/滚动切数据）
        train_start: str = None,
        train_end: str = None,
        valid_start: str = None,
        valid_end: str = None,
        test_start: str = None,
        test_end: str = None,
        # 性能选项：若输出文件已存在则跳过生成（适合滚动/增量反复重跑）
        skip_if_exists: bool = False,
):

    t0 = time.perf_counter()
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

    print(f"[data_generator] folder_name={folder_name}")
    print(f"[data_generator] using config: {cfg_path}")

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

    try:
        mk = config["task"]["model"]["kwargs"]
        print(f"[data_generator] model dims: d_feat={mk.get('d_feat')}, d_model={mk.get('d_model')}, step_len={config['task']['dataset']['kwargs'].get('step_len')}")
    except Exception:
        # 不影响主流程
        pass

    start_date = config["task"]["dataset"]["kwargs"]["segments"]["train"][0].strftime("%Y%m%d")
    end_date = config["task"]["dataset"]["kwargs"]["segments"]["test"][1].strftime("%Y%m%d")
    train_seg = config["task"]["dataset"]["kwargs"]["segments"]["train"]
    valid_seg = config["task"]["dataset"]["kwargs"]["segments"]["valid"]
    test_seg = config["task"]["dataset"]["kwargs"]["segments"]["test"]
    print(f"[data_generator] segments train={train_seg[0]}~{train_seg[1]} valid={valid_seg[0]}~{valid_seg[1]} test={test_seg[0]}~{test_seg[1]}")

    # 若本轮数据已生成，则直接返回（避免重复跑 dataset.prepare + 大文件序列化）
    dl_train_path = Path(f'{save_path}/{market_name}_self_dl_train.pkl')
    dl_valid_path = Path(f'{save_path}/{market_name}_self_dl_valid.pkl')
    dl_test_path = Path(f'{save_path}/{market_name}_self_dl_test.pkl')
    if bool(skip_if_exists) and dl_train_path.exists() and dl_valid_path.exists() and dl_test_path.exists():
        # 仅在缺失时补齐本地 config，避免“跳过时覆盖已有 config”导致不可复现
        if not Path(local_config_path).exists():
            with open(local_config_path, "w") as f:
                yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
        print("[data_generator] skip_if_exists=True 且输出已存在，跳过数据生成。")
        print(f"[data_generator] existing: {dl_train_path.name}, {dl_valid_path.name}, {dl_test_path.name}")
        return

    # 将本次实际使用的 config 落盘到实验目录，保证实验可复现（避免依赖外部默认 config）
    with open(local_config_path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)

    h_conf = config["task"]["dataset"]["kwargs"]["handler"]
    # 避免“全量/轻量”handler 文件互相覆盖：文件名区分模式
    handler_tag = "full" if bool(handler_dump_all) else "light"
    h_path = Path(f'{save_path}/' + f'handler_{start_date}_{end_date}_{handler_tag}.pkl')

    if not h_path.exists():
        # 关键优化：仅为了写一个可复用的 handler 配置文件时，不要在这里 init_data=True
        # 否则会提前加载/处理大规模数据，并在 dump_all=True 时把私有数据全部序列化，极易 OOM。
        print(f"[data_generator] building handler stub: {h_path} (dump_all={bool(handler_dump_all)})")
        if isinstance(h_conf, dict):
            # 在不破坏原 config 的前提下插入 init_data=False（Alpha158/DataHandlerLP 支持通过 kwargs 透传）
            h_conf = dict(h_conf)
            h_conf_kwargs = dict(h_conf.get("kwargs", {}) or {})
            h_conf_kwargs.setdefault("init_data", False)
            h_conf["kwargs"] = h_conf_kwargs
        h = init_instance_by_config(h_conf)
        h.to_pickle(h_path, dump_all=bool(handler_dump_all))
        print(f"[data_generator] handler saved: {h_path}")
        # print('Save preprocessed data to', h_path)
    else:
        print(f"[data_generator] reuse handler: {h_path}")

    # Dataset 初始化时（尤其是 TSDatasetH/MASTERTSDatasetH）会立刻调用 handler.fetch 来生成 calendar；
    # 因此当 handler_dump_all=False（轻量 handler 不含 _data/_infer/_learn）时，必须触发 handler.setup_data。
    config["task"]["dataset"]["kwargs"]["handler"] = f"file://{h_path}"
    if not bool(handler_dump_all):
        config["task"]["dataset"]["kwargs"]["handler_kwargs"] = {}
    dataset = init_instance_by_config(config['task']["dataset"])
    print("[data_generator] preparing dataset (train/valid/test)...")
    dl_train = dataset.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
    dl_valid = dataset.prepare("valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
    dl_test = dataset.prepare("test", col_set=["feature", "label"], data_key=DataHandlerLP.DK_I)
    t_prepare = time.perf_counter()
    print(f"[data_generator] dataset.prepare done, elapsed={t_prepare - t0:.2f}s")

    print("[data_generator] dumping dl_train/dl_valid/dl_test pkl...")
    # 使用最高协议：通常更快且文件更小
    with open(dl_train_path, 'wb') as file: pickle.dump(dl_train, file, protocol=pickle.HIGHEST_PROTOCOL)
    with open(dl_valid_path, 'wb') as file: pickle.dump(dl_valid, file, protocol=pickle.HIGHEST_PROTOCOL)
    with open(dl_test_path, 'wb') as file: pickle.dump(dl_test, file, protocol=pickle.HIGHEST_PROTOCOL)
    t_dump = time.perf_counter()
    print(f"[data_generator] dump done, elapsed={t_dump - t_prepare:.2f}s (total={t_dump - t0:.2f}s)")
    print("[data_generator] done.")


if __name__ == "__main__":
    fire.Fire(data_generator)