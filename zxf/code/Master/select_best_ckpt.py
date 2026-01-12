"""
select_best_ckpt.py

在上一轮实验目录中，评估每个 seed 的“最后一个 checkpoint”的 valid 指标，
并选出单最优（默认 valid_IC 最大）作为 warm-start 的起点。

说明：
- 评估逻辑复用同目录的 offline_eval.py（加载 ckpt -> model.predict -> 计算 IC/ICIR/RIC/RICIR）
- 默认只评估 scope=last（每 seed 最大 epoch）
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import fire
import pandas as pd


# 允许在任意工作目录运行：确保能 import 到同目录下的 offline_eval.py
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from offline_eval import offline_eval  # noqa: E402


def _metric_col(metric: str) -> str:
    m = str(metric).strip().lower()
    if m in {"ic", "valid_ic"}:
        return "IC"
    if m in {"icir", "valid_icir"}:
        return "ICIR"
    if m in {"ric", "valid_ric"}:
        return "RIC"
    if m in {"ricir", "valid_ricir"}:
        return "RICIR"
    raise ValueError("metric 仅支持: IC | ICIR | RIC | RICIR（或带 valid_ 前缀）")


def select_best_ckpt(
    market_name: str = "csi800",
    prev_folder_name: str = "csi800_prev_exp_folder",
    data_path: str = "/home/idc2/notebook/zxf/data",
    split: str = "valid",
    ckpt_subdir: str = "Master_results",
    scope: str = "last",
    metric: str = "valid_IC",
    force_eval: bool = False,
    gpu: Optional[int] = None,
    enable_rank_loss: bool = False,
    out_json: Optional[str] = None,
) -> str:
    """
    选出 best checkpoint，并输出 json 元信息。

    - prev_folder_name: 上一轮实验目录名（位于 {data_path}/master_results/{prev_folder_name}）
    - metric: 默认 valid_IC（最大者）
    - force_eval: False 时若发现 offline_eval 生成的 csv 已存在，则直接读取复用
    """
    data_path = os.path.abspath(os.path.expanduser(data_path))
    exp_dir = os.path.join(data_path, "master_results", prev_folder_name)
    ckpt_dir = os.path.join(exp_dir, ckpt_subdir)
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"找不到 ckpt_dir: {ckpt_dir}")

    split = str(split).strip().lower()
    scope = str(scope).strip().lower()
    mcol = _metric_col(metric)

    # offline_eval 默认输出路径（与 offline_eval.py 一致）
    offline_csv = os.path.join(ckpt_dir, f"offline_{split}_metrics_{scope}.csv")

    if (not force_eval) and os.path.exists(offline_csv):
        df = pd.read_csv(offline_csv)
    else:
        df = offline_eval(
            market_name=market_name,
            folder_name=prev_folder_name,
            data_path=data_path,
            split=split,
            ckpt_subdir=ckpt_subdir,
            scope=scope,
            gpu=gpu,
            enable_rank_loss=enable_rank_loss,
            out_csv=offline_csv,
            return_df=True,
            progress=True,
            log_every=50,
        )

    if df is None or len(df) == 0:
        raise RuntimeError("offline metrics 为空，无法选择 best ckpt")

    if mcol not in df.columns:
        raise KeyError(f"离线指标缺少列 {mcol}，实际列={list(df.columns)}")

    # 取 metric 最大者；如有并列，按 seed/epoch 取更大的 epoch（更“新”）
    df2 = df.copy()
    df2[mcol] = pd.to_numeric(df2[mcol], errors="coerce")
    df2 = df2.dropna(subset=[mcol])
    if len(df2) == 0:
        raise RuntimeError(f"列 {mcol} 全为空/不可解析，无法选择 best ckpt")

    df2 = df2.sort_values([mcol, "epoch"], ascending=[False, False]).reset_index(drop=True)
    best = df2.iloc[0].to_dict()

    # 输出元信息
    now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    meta: Dict[str, Any] = {
        "prev_folder_name": prev_folder_name,
        "market_name": market_name,
        "split": split,
        "scope": scope,
        "metric": mcol,
        "selected": {
            "seed": int(best.get("seed")),
            "epoch": int(best.get("epoch")),
            "ckpt_file": str(best.get("ckpt_file")),
            "ckpt_dir": ckpt_dir,
            "ckpt_path": os.path.join(ckpt_dir, str(best.get("ckpt_file"))),
            "IC": float(best.get("IC")),
            "ICIR": float(best.get("ICIR")),
            "RIC": float(best.get("RIC")),
            "RICIR": float(best.get("RICIR")),
        },
        "generated_at": now,
        "offline_metrics_csv": offline_csv,
    }

    if out_json is None or str(out_json).strip() == "":
        out_json = os.path.join(exp_dir, f"best_ckpt_{split}_{mcol.lower()}.json")
    out_path = Path(out_json).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[SUCCESS] best ckpt meta written:", str(out_path))
    print("[INFO] selected:", meta["selected"])
    return str(out_path)


if __name__ == "__main__":
    fire.Fire(select_best_ckpt)

