"""
roll_config.py

基于 Qlib 交易日历，把上一轮 config 的 train/valid/test 窗口整体滚动到“最新可用交易日”，
并保持每段长度（交易日数）与段间 gap（交易日数）不变。

典型用法（增量训练实验）：
1) 从 prev_folder_name 读取上一轮实验目录中的 workflow_config
2) 用 provider_uri 的最新交易日计算新窗口边界
3) 写出新 workflow_config 到 out_folder_name 对应的实验目录
"""

from __future__ import annotations

import datetime as _dt
import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import fire
import pandas as pd
import yaml


# 将本地 qlib 源码目录设置为最高优先级（仓库内自带）
QLIB_DIRNAME = "/home/idc2/notebook/qlib"
if QLIB_DIRNAME not in sys.path:
    sys.path.insert(0, QLIB_DIRNAME)

import qlib  # noqa: E402
from qlib.constant import REG_CN  # noqa: E402
from qlib.data import D  # noqa: E402


def _to_timestamp(x: Any) -> pd.Timestamp:
    if isinstance(x, pd.Timestamp):
        return x
    if isinstance(x, (_dt.datetime, _dt.date)):
        return pd.Timestamp(x)
    return pd.Timestamp(str(x))


def _align_start(cal: pd.DatetimeIndex, dt: Any) -> int:
    """对齐到 >=dt 的第一个交易日索引"""
    ts = _to_timestamp(dt)
    i = int(cal.searchsorted(ts, side="left"))
    if i >= len(cal):
        raise ValueError(f"start 越界：{dt} 超出交易日历范围（latest={cal[-1].date()}）")
    return i


def _align_end(cal: pd.DatetimeIndex, dt: Any) -> int:
    """对齐到 <=dt 的最后一个交易日索引"""
    ts = _to_timestamp(dt)
    i = int(cal.searchsorted(ts, side="right")) - 1
    if i < 0:
        raise ValueError(f"end 越界：{dt} 早于交易日历起点（earliest={cal[0].date()}）")
    return i


def _seg_len(cal: pd.DatetimeIndex, start: Any, end: Any) -> Tuple[int, int, int]:
    """
    返回 (start_i, end_i, length)。
    - start_i: 对齐后的 start trading index
    - end_i  : 对齐后的 end trading index
    - length : 交易日长度（含两端）
    """
    s_i = _align_start(cal, start)
    e_i = _align_end(cal, end)
    if e_i < s_i:
        raise ValueError(f"段非法：start={start} end={end}（对齐后 start_i={s_i} end_i={e_i}）")
    return s_i, e_i, int(e_i - s_i + 1)


def _date_at(cal: pd.DatetimeIndex, i: int) -> _dt.date:
    return pd.Timestamp(cal[int(i)]).date()


def _get_segments(cfg: Dict[str, Any]) -> Dict[str, Tuple[Any, Any]]:
    seg = cfg["task"]["dataset"]["kwargs"]["segments"]
    out: Dict[str, Tuple[Any, Any]] = {}
    for k in ("train", "valid", "test"):
        v = seg[k]
        if isinstance(v, (list, tuple)) and len(v) == 2:
            out[k] = (v[0], v[1])
        else:
            raise ValueError(f"segments[{k}] 期望为长度=2 的 list/tuple，实际={v}")
    return out


def _update_if_exists(d: Dict[str, Any], path: Tuple[str, ...], value: Any) -> None:
    cur: Any = d
    for k in path[:-1]:
        if not isinstance(cur, dict) or k not in cur:
            return
        cur = cur[k]
    if isinstance(cur, dict) and path[-1] in cur:
        cur[path[-1]] = value


def roll_config(
    market_name: str = "csi800",
    data_path: str = "/home/idc2/notebook/zxf/data",
    provider_uri: str = "/home/idc2/notebook/qlib_bin/cn_data_train",
    prev_folder_name: str | None = None,
    prev_config_path: str | None = None,
    out_folder_name: str | None = None,
    out_config_path: str | None = None,
    region: str = "cn",
    freq: str = "day",
    dry_run: bool = False,
) -> str:
    """
    生成滚动后的 workflow_config，并写入文件。

    读取优先级：
    - prev_config_path（显式指定）
    - data_path/master_results/{prev_folder_name}/workflow_config_master_Alpha158_{market}.yaml
    - data_path/workflow_config_master_Alpha158_{market}.yaml
    """

    # -------- 1) load config --------
    data_path = os.path.abspath(os.path.expanduser(data_path))
    if prev_config_path:
        cfg_path = Path(prev_config_path).expanduser().resolve()
    else:
        if prev_folder_name:
            cand = Path(data_path) / "master_results" / prev_folder_name / f"workflow_config_master_Alpha158_{market_name}.yaml"
            if cand.exists():
                cfg_path = cand
            else:
                cfg_path = Path(data_path) / f"workflow_config_master_Alpha158_{market_name}.yaml"
        else:
            cfg_path = Path(data_path) / f"workflow_config_master_Alpha158_{market_name}.yaml"

    if not cfg_path.exists():
        raise FileNotFoundError(f"找不到 config: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)

    segs = _get_segments(cfg)

    # -------- 2) init qlib / calendar --------
    region = str(region).strip().lower()
    if region != "cn":
        raise ValueError("当前仅支持 region='cn'（如需扩展可再补）")

    qlib.init(provider_uri=provider_uri, region=REG_CN)
    cal = pd.DatetimeIndex(D.calendar(freq=freq))
    if len(cal) < 10:
        raise RuntimeError("交易日历长度异常，检查 provider_uri 是否正确")
    latest_i = len(cal) - 1

    # -------- 3) compute lengths & gaps (trading days) --------
    tr_s_i, tr_e_i, L_tr = _seg_len(cal, *segs["train"])
    va_s_i, va_e_i, L_va = _seg_len(cal, *segs["valid"])
    te_s_i, te_e_i, L_te = _seg_len(cal, *segs["test"])

    G_tr_va = int(va_s_i - tr_e_i - 1)
    G_va_te = int(te_s_i - va_e_i - 1)
    if G_tr_va < 0 or G_va_te < 0:
        raise ValueError(
            f"segments gap 非法（train/valid/test 可能重叠或顺序错误）：G_tr_va={G_tr_va}, G_va_te={G_va_te}"
        )

    # -------- 4) roll to latest (keep L & G) --------
    new_te_e_i = latest_i
    new_te_s_i = new_te_e_i - (L_te - 1)
    new_va_e_i = new_te_s_i - (G_va_te + 1)
    new_va_s_i = new_va_e_i - (L_va - 1)
    new_tr_e_i = new_va_s_i - (G_tr_va + 1)
    new_tr_s_i = new_tr_e_i - (L_tr - 1)
    if new_tr_s_i < 0:
        raise ValueError(
            "滚动后 train_start 越界（历史窗口太长或 latest 太早）。"
            f"new_tr_s_i={new_tr_s_i}, earliest={cal[0].date()}, latest={cal[-1].date()}"
        )

    new_train = (_date_at(cal, new_tr_s_i), _date_at(cal, new_tr_e_i))
    new_valid = (_date_at(cal, new_va_s_i), _date_at(cal, new_va_e_i))
    new_test = (_date_at(cal, new_te_s_i), _date_at(cal, new_te_e_i))

    # -------- 5) update cfg fields --------
    cfg["task"]["dataset"]["kwargs"]["segments"]["train"] = list(new_train)
    cfg["task"]["dataset"]["kwargs"]["segments"]["valid"] = list(new_valid)
    cfg["task"]["dataset"]["kwargs"]["segments"]["test"] = list(new_test)

    # 常见字段同步更新（存在则更新；不存在则跳过）
    _update_if_exists(cfg, ("train_date_start",), new_train[0])
    _update_if_exists(cfg, ("train_date_end",), new_train[1])
    _update_if_exists(cfg, ("valid_date_start",), new_valid[0])
    _update_if_exists(cfg, ("valid_date_end",), new_valid[1])
    _update_if_exists(cfg, ("test_date_start",), new_test[0])
    _update_if_exists(cfg, ("test_date_end",), new_test[1])

    # handler config：start/end/fit_end 同步
    for base_key in ("data_handler_config", "market_data_handler_config"):
        if base_key in cfg and isinstance(cfg[base_key], dict):
            cfg[base_key]["start_time"] = new_train[0]
            cfg[base_key]["end_time"] = new_test[1]
            cfg[base_key]["fit_start_time"] = new_train[0]
            cfg[base_key]["fit_end_time"] = new_train[1]

    # dataset handler kwargs 也尽量同步（有些 yaml 没有 data_handler_config 顶层）
    try:
        hkw = cfg["task"]["dataset"]["kwargs"]["handler"]["kwargs"]
        if isinstance(hkw, dict):
            hkw["start_time"] = new_train[0]
            hkw["end_time"] = new_test[1]
            hkw["fit_start_time"] = new_train[0]
            hkw["fit_end_time"] = new_train[1]
    except Exception:
        pass

    # port_analysis backtest（如存在）
    try:
        bt = cfg["port_analysis_config"]["backtest"]
        if isinstance(bt, dict):
            bt["start_time"] = new_test[0]
            bt["end_time"] = new_test[1]
    except Exception:
        pass

    # qlib_init（如存在）
    if "qlib_init" in cfg and isinstance(cfg["qlib_init"], dict):
        cfg["qlib_init"]["provider_uri"] = provider_uri

    # -------- 6) resolve output path --------
    if out_config_path:
        out_path = Path(out_config_path).expanduser().resolve()
    else:
        if not out_folder_name:
            raise ValueError("未指定 out_config_path 时，必须提供 out_folder_name")
        exp_dir = Path(data_path) / "master_results" / out_folder_name
        out_path = exp_dir / f"workflow_config_master_Alpha158_{market_name}.yaml"

    if dry_run:
        print("[DRY_RUN] cfg_path:", str(cfg_path))
        print("[DRY_RUN] provider_uri:", provider_uri)
        print("[DRY_RUN] latest:", cal[-1].date())
        print("[DRY_RUN] old train/valid/test:", segs)
        print("[DRY_RUN] new train/valid/test:", {"train": new_train, "valid": new_valid, "test": new_test})
        return str(out_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    print("[SUCCESS] Rolled config written:", str(out_path))
    print("[INFO] latest trading day:", cal[-1].date())
    print("[INFO] new train/valid/test:", {"train": new_train, "valid": new_valid, "test": new_test})
    return str(out_path)


if __name__ == "__main__":
    fire.Fire(roll_config)

