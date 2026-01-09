import os
import re
import sys
import pickle
from typing import Dict, List, Optional, Tuple

import fire
import pandas as pd
import yaml


# 允许在任意工作目录运行：确保能 import 到同目录下的 master.py
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from master import MASTERModel  # noqa: E402


def _load_config(experimental_data_path: str, market_name: str) -> dict:
    cfg_path = os.path.join(
        experimental_data_path, f"workflow_config_master_Alpha158_{market_name}.yaml"
    )
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def _load_dl(experimental_data_path: str, universe: str, split: str):
    pkl_path = os.path.join(experimental_data_path, f"{universe}_self_dl_{split}.pkl")
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def _parse_ckpt_filename(fname: str) -> Optional[Tuple[int, int]]:
    """
    解析 checkpoint 文件名，返回 (seed, epoch)。
    期望格式：..._self_exp_{seed}_{epoch}.pkl
    """
    if not fname.endswith(".pkl"):
        return None
    stem = fname[:-4]
    if "_" not in stem:
        return None
    prefix, epoch_str = stem.rsplit("_", 1)
    try:
        epoch = int(epoch_str)
    except Exception:
        return None
    m = re.search(r"_self_exp_(\d+)$", prefix)
    if m is None:
        return None
    seed = int(m.group(1))
    return seed, epoch


def _select_ckpts(
    ckpt_dir: str,
    seeds: Optional[List[int]] = None,
    scope: str = "last",
    every_k: int = 1,
    min_epoch: Optional[int] = None,
    max_epoch: Optional[int] = None,
) -> List[dict]:
    scope = str(scope).strip().lower()
    if scope not in {"last", "all", "every"}:
        raise ValueError("scope 仅支持: 'last' | 'all' | 'every'")
    if scope == "every" and int(every_k) <= 0:
        raise ValueError("scope='every' 时 every_k 必须为正整数")

    wanted_seeds = None
    if seeds is not None:
        wanted_seeds = set(int(x) for x in seeds)

    recs: List[dict] = []
    for fname in os.listdir(ckpt_dir):
        parsed = _parse_ckpt_filename(fname)
        if parsed is None:
            continue
        seed, epoch = parsed
        if wanted_seeds is not None and seed not in wanted_seeds:
            continue
        if min_epoch is not None and epoch < int(min_epoch):
            continue
        if max_epoch is not None and epoch > int(max_epoch):
            continue
        recs.append(
            {
                "seed": seed,
                "epoch": epoch,
                "ckpt_file": fname,
                "ckpt_path": os.path.join(ckpt_dir, fname),
            }
        )

    if len(recs) == 0:
        return []

    # 排序：seed, epoch 递增
    recs.sort(key=lambda x: (x["seed"], x["epoch"]))

    if scope == "all":
        return recs

    if scope == "every":
        k = int(every_k)
        return [r for r in recs if (int(r["epoch"]) % k == 0)]

    # scope == "last": 每个 seed 取最大 epoch
    last: Dict[int, dict] = {}
    for r in recs:
        s = int(r["seed"])
        if s not in last or int(r["epoch"]) > int(last[s]["epoch"]):
            last[s] = r
    out = list(last.values())
    out.sort(key=lambda x: (x["seed"], x["epoch"]))
    return out


def offline_eval(
    market_name: str = "csi800",
    folder_name: str = "csi800_20260107_f8_20150101_20251231",
    data_path: str = "/home/idc2/notebook/zxf/data",
    split: str = "valid",
    ckpt_subdir: str = "Master_results",
    scope: str = "all",
    every_k: int = 10,
    seeds: Optional[List[int]] = None,
    min_epoch: Optional[int] = None,
    max_epoch: Optional[int] = None,
    gpu: Optional[int] = None,
    enable_rank_loss: bool = False,
    out_csv: Optional[str] = None,
    return_df: bool = False,
):
    """
    训练后“后算”验证/测试指标：
    - 训练时设 MASTER_EVAL_FREQ=0（或更大）避免每个 epoch 卡在 CPU 指标计算上
    - 训练结束后，用本脚本加载 checkpoint 再计算 IC/ICIR/RIC/RICIR

    参数说明（重点）：
    - split: 'valid' 或 'test'
    - scope:
        - 'last'  : 每个 seed 只评估最后一个 epoch（最快，默认）
        - 'every' : 每 every_k 个 epoch 评估一次
        - 'all'   : 评估所有 checkpoint（最慢）
    """
    split = str(split).strip().lower()
    if split not in {"valid", "test"}:
        raise ValueError("split 仅支持: 'valid' | 'test'")

    experimental_data_path = os.path.join(data_path, "master_results", folder_name)
    cfg = _load_config(experimental_data_path, market_name)
    universe = cfg["market"]

    # 从 config 读取模型超参（与 Master/main.py 保持一致）
    mkw = cfg["task"]["model"]["kwargs"]
    d_feat = mkw["d_feat"]
    d_model = mkw["d_model"]
    t_nhead = mkw["t_nhead"]
    s_nhead = mkw["s_nhead"]
    dropout = mkw["dropout"]
    gate_input_start_index = mkw["gate_input_start_index"]
    gate_input_end_index = mkw["gate_input_end_index"]
    lr = mkw["lr"]
    train_stop_loss_thred = mkw["train_stop_loss_thred"]
    beta = mkw["beta"]
    GPU = int(mkw["GPU"]) if gpu is None else int(gpu)

    ckpt_dir = os.path.join(experimental_data_path, ckpt_subdir)
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"找不到 checkpoint 目录: {ckpt_dir}")

    dl = _load_dl(experimental_data_path, universe, split)

    ckpts = _select_ckpts(
        ckpt_dir=ckpt_dir,
        seeds=seeds,
        scope=scope,
        every_k=every_k,
        min_epoch=min_epoch,
        max_epoch=max_epoch,
    )
    if len(ckpts) == 0:
        raise FileNotFoundError("未找到符合条件的 checkpoint（请检查 folder_name/ckpt_subdir/seeds/scope）。")

    # 每个 seed 复用一个模型实例，只更新参数（state_dict）
    models: Dict[int, MASTERModel] = {}
    rows: List[dict] = []

    for r in ckpts:
        seed = int(r["seed"])
        epoch = int(r["epoch"])
        ckpt_path = r["ckpt_path"]

        if seed not in models:
            models[seed] = MASTERModel(
                d_feat=d_feat,
                d_model=d_model,
                t_nhead=t_nhead,
                s_nhead=s_nhead,
                T_dropout_rate=dropout,
                S_dropout_rate=dropout,
                beta=beta,
                gate_input_end_index=gate_input_end_index,
                gate_input_start_index=gate_input_start_index,
                n_epochs=1,  # 离线评估不训练
                lr=lr,
                GPU=GPU,
                seed=seed,
                train_stop_loss_thred=train_stop_loss_thred,
                save_path=ckpt_dir,
                save_prefix="",
                enable_rank_loss=enable_rank_loss,
                eval_freq=0,
            )

        model = models[seed]
        model.load_param(ckpt_path)
        model.fitted = epoch
        _, metrics = model.predict(dl)
        rows.append(
            {
                "seed": seed,
                "epoch": epoch,
                "split": split,
                "ckpt_file": os.path.basename(ckpt_path),
                "IC": metrics["IC"],
                "ICIR": metrics["ICIR"],
                "RIC": metrics["RIC"],
                "RICIR": metrics["RICIR"],
            }
        )

    df = pd.DataFrame(rows).sort_values(["seed", "epoch"]).reset_index(drop=True)
    if out_csv is None or str(out_csv).strip() == "":
        out_csv = os.path.join(ckpt_dir, f"offline_{split}_metrics_{scope}.csv")
    df.to_csv(out_csv, index=False)
    print(f"[SUCCESS] 离线指标已保存: {out_csv}")
    # 说明：
    # - 这里默认不返回 DataFrame（否则 python-fire 可能会进入 DataFrame 的“帮助页/分页器”，需要手动按 q 退出）
    # - 如确实需要返回 DataFrame（比如在交互环境里），设置 return_df=True
    return df if bool(return_df) else out_csv


if __name__ == "__main__":
    fire.Fire(offline_eval)


