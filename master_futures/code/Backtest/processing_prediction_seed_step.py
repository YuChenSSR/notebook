"""
processing_prediction_seed_step.py

期货版：从 MASTER 训练输出的 checkpoint（*_self_exp_{seed}_{step}.pkl）生成预测文件，并落盘到：
  {data_path}/master_results/{folder_name}/Backtest_Results/predictions/

调用方式对齐股票版 master_oneclick.sh：
  cd master_futures/code/Backtest
  python processing_prediction_seed_step.py --data_path=... --market_name=f88 --folder_name=...
"""

from __future__ import annotations

import os
import sys
import pickle
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import fire
import pandas as pd
import yaml


# 将本地 qlib 源码目录设置为最高优先级（仓库内自带）
QLIB_DIRNAME = "/home/idc2/notebook/qlib"
if QLIB_DIRNAME not in sys.path:
    sys.path.insert(0, QLIB_DIRNAME)

# 确保 repo root 可 import（用于 module_path: master_futures.*）
_THIS_FILE = Path(__file__).resolve()
REPO_ROOT = str(_THIS_FILE.parents[3])  # .../notebook
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


from master_futures.code.Master.master import MASTERModel  # noqa: E402


def _parse_seed_step_from_ckpt(filename: str) -> Tuple[int, int]:
    """
    从 checkpoint 文件名末尾提取 seed 与 step。

    例如：
    - f88_backday_8_self_exp_46_0.pkl
    - f88_rank_backday_8_self_exp_46_10.pkl
    """
    stem = filename.rsplit(".", 1)[0]
    parts = stem.split("_")
    if len(parts) < 2:
        raise ValueError(f"Invalid ckpt filename: {filename}")
    seed_str, step_str = parts[-2], parts[-1]
    if not (seed_str.isdigit() and step_str.isdigit()):
        raise ValueError(f"Cannot parse seed/step from ckpt filename: {filename}")
    return int(seed_str), int(step_str)


def _load_config(cfg_path: str) -> dict:
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_dl_test(exp_dir: str, universe: str):
    p = os.path.join(exp_dir, f"{universe}_self_dl_test.pkl")
    with open(p, "rb") as f:
        return pickle.load(f)


def _instantiate_model(cfg: dict, seed: int):
    mkw = cfg["task"]["model"]["kwargs"]
    d_feat = int(mkw["d_feat"])
    d_model = int(mkw["d_model"])
    t_nhead = int(mkw["t_nhead"])
    s_nhead = int(mkw["s_nhead"])
    dropout = float(mkw["dropout"])
    gate_input_start_index = int(mkw["gate_input_start_index"])
    gate_input_end_index = int(mkw["gate_input_end_index"])
    lr = float(mkw["lr"])
    train_stop_loss_thred = float(mkw["train_stop_loss_thred"])
    beta = float(mkw["beta"])
    GPU = int(mkw.get("GPU", 0))

    # 这里只用于推理，不训练：n_epochs=1 即可；eval_freq=0 避免额外开销
    return MASTERModel(
        d_feat=d_feat,
        d_model=d_model,
        t_nhead=t_nhead,
        s_nhead=s_nhead,
        T_dropout_rate=dropout,
        S_dropout_rate=dropout,
        beta=beta,
        gate_input_end_index=gate_input_end_index,
        gate_input_start_index=gate_input_start_index,
        n_epochs=1,
        lr=lr,
        GPU=GPU,
        seed=int(seed),
        train_stop_loss_thred=train_stop_loss_thred,
        save_path="",   # 推理不保存 ckpt
        save_prefix="",
        eval_freq=0,
    )


def _select_ckpts(ckpt_dir: Path, scope: str = "all") -> List[Path]:
    """
    - scope=all: 处理所有 ckpt（对齐股票版）
    - scope=last: 每个 seed 只取最大 step
    """
    scope = str(scope).strip().lower()
    ckpts = sorted([p for p in ckpt_dir.glob("*self_exp*.pkl") if p.is_file()])
    if scope == "all":
        return ckpts
    if scope != "last":
        raise ValueError("scope 仅支持 all/last")

    # last：每个 seed 取最大 step
    best: Dict[int, Tuple[int, Path]] = {}
    for p in ckpts:
        seed, step = _parse_seed_step_from_ckpt(p.name)
        if seed not in best or step > best[seed][0]:
            best[seed] = (step, p)
    out = [v[1] for v in best.values()]
    out.sort(key=lambda x: _parse_seed_step_from_ckpt(x.name))
    return out


def main(
    data_path: str = "/home/idc2/notebook/master_futures/data",
    market_name: str = "f88",
    folder_name: str = "f88_20260113_20100104_20251212",
    scope: str = "all",
    overwrite: int = 1,
) -> str:
    """
    生成预测文件到 Backtest_Results/predictions，并写 test_metrics_results.csv。
    返回 predictions 目录路径。
    """
    data_path = os.path.abspath(os.path.expanduser(data_path))
    exp_dir = os.path.join(data_path, "master_results", folder_name)
    cfg_path = os.path.join(exp_dir, f"workflow_config_master_Alpha158_{market_name}.yaml")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"找不到实验 config：{cfg_path}")
    cfg = _load_config(cfg_path)

    universe = str(cfg.get("market", market_name))
    backday = int(cfg["task"]["dataset"]["kwargs"]["step_len"])

    dl_test = _load_dl_test(exp_dir=exp_dir, universe=universe)

    ckpt_dir = Path(exp_dir) / "Master_results"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"找不到 ckpt 目录：{ckpt_dir}")

    ckpts = _select_ckpts(ckpt_dir=ckpt_dir, scope=scope)
    if len(ckpts) == 0:
        raise FileNotFoundError(f"未找到 ckpt：{ckpt_dir}/*self_exp*.pkl")

    out_dir = Path(exp_dir) / "Backtest_Results" / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    overwrite = bool(int(overwrite))

    for ckpt in ckpts:
        seed, step = _parse_seed_step_from_ckpt(ckpt.name)
        out_csv = out_dir / f"master_predictions_backday_{backday}_{universe}_{seed}_{step}.csv"
        if out_csv.exists() and (not overwrite):
            print(f"[SKIP] exists: {out_csv}")
            continue

        model = _instantiate_model(cfg, seed=seed)
        model.load_param(str(ckpt))
        model.fitted = int(step)

        predictions, metrics = model.predict(dl_test)

        pred_frame = predictions.to_frame()
        pred_frame.columns = ["score"]
        pred_frame.reset_index(inplace=True)
        pred_frame.to_csv(out_csv, index=False, date_format="%Y-%m-%d")
        print(f"[OK] wrote: {out_csv}")

        rows.append(
            {
                "Seed": int(seed),
                "Step": int(step),
                "Test_IC": float(metrics.get("IC", float("nan"))),
                "Test_ICIR": float(metrics.get("ICIR", float("nan"))),
                "Test_RIC": float(metrics.get("RIC", float("nan"))),
                "Test_RICIR": float(metrics.get("RICIR", float("nan"))),
            }
        )

    if len(rows) > 0:
        df = pd.DataFrame(rows).sort_values(["Seed", "Step"]).reset_index(drop=True)
        df.to_csv(out_dir / "test_metrics_results.csv", index=False)

    return str(out_dir)


if __name__ == "__main__":
    fire.Fire(main)

