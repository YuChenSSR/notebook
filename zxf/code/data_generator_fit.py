import sys
import os
import yaml
import fire
import gc
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# =========================
# Qlib Path
# =========================
QLIB_DIRNAME = '/home/idc2/notebook/qlib'
sys.path.insert(0, QLIB_DIRNAME)

import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config
from qlib.data.dataset.handler import DataHandlerLP


# =========================
# Parquet 追加写入
# =========================
def append_parquet(df: pd.DataFrame, parquet_path: str):
    table = pa.Table.from_pandas(df, preserve_index=True)

    if not os.path.exists(parquet_path):
        pq.write_table(table, parquet_path)
    else:
        pq.write_table(
            table,
            parquet_path,
            append=True
        )


# =========================
# 主流程
# =========================
def data_generator(
    market_name: str = "csi800b",
    qlib_path: str = "/home/idc2/notebook/qlib_bin/cn_data_train",
    data_path: str = "/home/idc2/notebook/zxf/data",
    folder_name: str = "csi800_parquet_data",
):

    # -------- 1. init qlib --------
    qlib.init(provider_uri=qlib_path, region=REG_CN)

    # -------- 2. load config --------
    with open(
        f"{data_path}/workflow_config_master_Alpha158_{market_name}.yaml", "r"
    ) as f:
        config = yaml.safe_load(f)

    save_path = f"{data_path}/master_results/{folder_name}"
    os.makedirs(save_path, exist_ok=True)

    # -------- 3. handler（轻量） --------
    h_conf = config["task"]["dataset"]["kwargs"]["handler"]
    h_path = Path(f"{save_path}/handler_light.pkl")

    if not h_path.exists():
        h = init_instance_by_config(h_conf)
        h.to_pickle(h_path, dump_all=False)
        del h
        gc.collect()

    config["task"]["dataset"]["kwargs"]["handler"] = f"file://{h_path}"
    dataset = init_instance_by_config(config["task"]["dataset"])

    # -------- 4. Train 分块 → 单 parquet --------
    train_parquet = f"{save_path}/{market_name}_train.parquet"
    if os.path.exists(train_parquet):
        os.remove(train_parquet)

    train_start, train_end = dataset.segments["train"]

    train_chunks = [
        (train_start, "20171231"),
        ("20180101", "20191231"),
        ("20200101", train_end),
    ]

    for i, (s, e) in enumerate(train_chunks):
        print(f"[INFO] Train chunk {i}: {s} → {e}")
        dataset.segments["train"] = (s, e)

        dl = dataset.prepare(
            "train",
            col_set=["feature", "label"],
            data_key=DataHandlerLP.DK_L
        )

        # 👉 核心：只取 data（DataFrame）
        df = dl.data

        append_parquet(df, train_parquet)

        del dl, df
        gc.collect()

    print("[SUCCESS] Train parquet generated")

    # -------- 5. Valid / Test（体量小，直接 parquet） --------
    for seg, key in [("valid", DataHandlerLP.DK_L), ("test", DataHandlerLP.DK_I)]:
        dl = dataset.prepare(seg, col_set=["feature", "label"], data_key=key)
        df = dl.data
        df.to_parquet(f"{save_path}/{market_name}_{seg}.parquet")
        del dl, df
        gc.collect()

    print("[SUCCESS] All data prepared safely")


if __name__ == "__main__":
    fire.Fire(data_generator)
