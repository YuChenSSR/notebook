"""
backtest_seed_step.py

期货版：读取 Backtest_Results/predictions 下的预测文件（master_predictions_backday_*_{seed}_{step}.csv），
用 Qlib TopkDropoutStrategy + SimulatorExecutor 做仿真回测，并输出：
  {data_path}/master_results/{folder_name}/Backtest_Results/results/backtest_result.csv

调用方式对齐股票版 master_oneclick.sh：
  cd master_futures/code/Backtest
  python backtest_seed_step.py --market_name=f88 --folder_name=...
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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


import qlib  # noqa: E402
from qlib.constant import REG_CN  # noqa: E402
from qlib.backtest import backtest as qlib_backtest  # noqa: E402
from qlib.contrib.evaluate import risk_analysis  # noqa: E402
from qlib.data.data import Cal  # noqa: E402


def _load_yaml(path: Union[str, Path]) -> dict:
    path = Path(path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _as_timestamp(x: Any) -> pd.Timestamp:
    return pd.Timestamp(x)


def _parse_seed_step_from_pred_filename(filename: str) -> Tuple[int, int]:
    """
    解析预测文件名末尾的 seed/step：
    - master_predictions_backday_8_f88_46_0.csv -> (46, 0)
    - master_predictions_backday_8_f88_46.csv   -> (46, -1)   # 兼容旧格式（无 step）
    """
    stem = filename.rsplit(".", 1)[0]
    parts = stem.split("_")
    if len(parts) < 1:
        raise ValueError(f"Invalid filename: {filename}")

    last = parts[-1]
    if not last.isdigit():
        raise ValueError(f"Cannot parse seed/step from filename: {filename}")

    # 末尾两段都是数字：seed_step
    if len(parts) >= 2 and parts[-2].isdigit():
        return int(parts[-2]), int(parts[-1])

    # 只有末尾一段是数字：仅 seed（step 记为 -1）
    return int(parts[-1]), -1


def _read_predictions_csv(path: Union[str, Path]) -> pd.Series:
    """
    读取预测文件，返回 MultiIndex(datetime, instrument) 的 score Series。
    """
    path = Path(path).expanduser().resolve()
    df = pd.read_csv(path)
    if not {"datetime", "instrument", "score"}.issubset(df.columns):
        raise ValueError(f"预测文件缺少列（需要 datetime/instrument/score）：{path} cols={list(df.columns)}")
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["instrument"] = df["instrument"].astype(str)
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df.dropna(subset=["score"])
    return df.set_index(["datetime", "instrument"])["score"].sort_index()


def _load_f88_codes_from_instruments_file(
    instruments_path: Union[str, Path],
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
) -> List[str]:
    instruments_path = Path(instruments_path).expanduser().resolve()
    if not instruments_path.exists():
        raise FileNotFoundError(f"找不到 instruments 文件：{instruments_path}")

    codes: List[str] = []
    with instruments_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            code, s, e = parts[0], parts[1], parts[2]
            s_ts = pd.Timestamp(s)
            e_ts = pd.Timestamp(e)
            if e_ts < start_time or s_ts > end_time:
                continue
            codes.append(str(code))
    if len(codes) == 0:
        raise ValueError(
            f"从 {instruments_path} 未筛到任何可用 benchmark 合约（start={start_time.date()} end={end_time.date()}）"
        )
    return codes


def _pick_benchmark(
    cfg: dict,
    qlib_path: str,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    instruments_file: Optional[str] = None,
    benchmark_override: Optional[str] = None,
) -> Union[str, List[str]]:
    if benchmark_override is not None and str(benchmark_override).strip() != "":
        return str(benchmark_override).strip()

    bt_cfg = (cfg.get("port_analysis_config", {}) or {}).get("backtest", {}) or {}
    bench = bt_cfg.get("benchmark", None)
    if bench is not None:
        if isinstance(bench, str) and bench.strip() != "":
            return bench.strip()
        if isinstance(bench, list) and len(bench) > 0:
            return [str(x) for x in bench]

    if instruments_file is None or str(instruments_file).strip() == "":
        instruments_file = os.path.join(os.path.expanduser(qlib_path), "instruments", "f88.txt")
    return _load_f88_codes_from_instruments_file(instruments_file, start_time=start_time, end_time=end_time)


def _safe_end_time_for_trade_calendar(end_time: pd.Timestamp, freq: str = "day") -> pd.Timestamp:
    """
    同 backtest_qlib.py：避免 end_time 落在日历最后一天导致 get_step_time 越界。
    """
    cal = pd.DatetimeIndex(Cal.calendar(freq=freq, future=True))
    if len(cal) < 3:
        raise RuntimeError(f"交易日历长度异常（len={len(cal)}），无法回测：freq={freq}")
    end_time = pd.Timestamp(end_time)
    i = int(cal.searchsorted(end_time, side="right") - 1)
    if i < 0:
        raise ValueError(f"end_time 早于交易日历起点：end_time={end_time.date()} earliest={cal[0].date()}")
    if i >= len(cal) - 1:
        i = len(cal) - 2
    return pd.Timestamp(cal[i])


def _analyze_report(report: pd.DataFrame) -> Dict[str, float]:
    ret_net = report["return"] - report.get("cost", 0.0)
    risk_ret = risk_analysis(ret_net)["risk"]
    out: Dict[str, float] = {
        "sharpe": float(risk_ret.loc["information_ratio"]),
        "annual_return": float(risk_ret.loc["annualized_return"]),
        "max_drawdown": float(risk_ret.loc["max_drawdown"]),
    }
    if "bench" in report.columns:
        excess_net = report["return"] - report["bench"] - report.get("cost", 0.0)
        risk_ex = risk_analysis(excess_net)["risk"]
        out.update(
            {
                "information_ratio": float(risk_ex.loc["information_ratio"]),
                "annual_excess_return": float(risk_ex.loc["annualized_return"]),
                "excess_max_drawdown": float(risk_ex.loc["max_drawdown"]),
            }
        )
    else:
        out.update(
            {
                "information_ratio": float("nan"),
                "annual_excess_return": float("nan"),
                "excess_max_drawdown": float("nan"),
            }
        )
    return out


def main(
    market_name: str = "f88",
    folder_name: str = "f88_20260113_20100104_20251212",
    data_path: str = "/home/idc2/notebook/master_futures/data",
    qlib_path: str = "/home/idc2/notebook/futures/data/qlib_bin/cn_data_backtest",
    instruments_file: str = "",
    benchmark_override: str = "",
    topk: Optional[int] = None,
    n_drop: Optional[int] = None,
    account: Optional[float] = None,
    freq: Optional[str] = None,
    deal_price: Optional[str] = None,
    open_cost: Optional[float] = None,
    close_cost: Optional[float] = None,
    min_cost: Optional[float] = None,
    limit_threshold: Optional[float] = None,
    save_report: Optional[int] = None,
) -> str:
    """
    批量回测 Backtest_Results/predictions 下的预测文件，输出 backtest_result.csv。
    返回：backtest_result.csv 的路径。
    """
    data_path = os.path.abspath(os.path.expanduser(data_path))
    exp_dir = os.path.join(data_path, "master_results", folder_name)
    cfg_path = os.path.join(exp_dir, f"workflow_config_master_Alpha158_{market_name}.yaml")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"找不到实验 config：{cfg_path}")
    cfg = _load_yaml(cfg_path)

    pred_dir = Path(exp_dir) / "Backtest_Results" / "predictions"
    if not pred_dir.exists():
        raise FileNotFoundError(f"找不到预测目录：{pred_dir}")
    pred_files = sorted([p for p in pred_dir.glob("master_predictions_backday_*.csv") if p.is_file()])
    if len(pred_files) == 0:
        raise FileNotFoundError(f"未找到预测文件：{pred_dir}/master_predictions_backday_*.csv")

    out_root = Path(exp_dir) / "Backtest_Results" / "results"
    out_root.mkdir(parents=True, exist_ok=True)
    out_csv = out_root / "backtest_result.csv"

    # Qlib init（回测用 provider_uri）
    qlib.init(provider_uri=os.path.expanduser(qlib_path), region=REG_CN)

    # 回测区间/参数：优先从 port_analysis_config 读取
    pa_cfg = cfg.get("port_analysis_config", {}) or {}
    st_cfg = pa_cfg.get("strategy", {}) or {}
    bt_cfg = pa_cfg.get("backtest", {}) or {}

    # strategy params（若命令行没传，则从 config 取；再没有就用默认）
    st_kwargs = (st_cfg.get("kwargs", {}) or {}) if isinstance(st_cfg, dict) else {}
    if topk is None:
        topk = int(st_kwargs.get("topk", 30))
    if n_drop is None:
        n_drop = int(st_kwargs.get("n_drop", 3))
    only_tradable = bool(st_kwargs.get("only_tradable", False))
    forbid_all_trade_at_limit = bool(st_kwargs.get("forbid_all_trade_at_limit", False))

    # backtest params（若命令行没传，则从 config 取；再没有就用默认）
    if freq is None:
        freq = str(bt_cfg.get("freq", "day"))
    if account is None:
        account = float(bt_cfg.get("account", 100_000_000))
    ex_kwargs_cfg = bt_cfg.get("exchange_kwargs", {}) or {}
    if deal_price is None:
        deal_price = str(ex_kwargs_cfg.get("deal_price", "close"))
    if limit_threshold is None:
        limit_threshold = float(ex_kwargs_cfg.get("limit_threshold", 0.095))
    if open_cost is None:
        open_cost = float(ex_kwargs_cfg.get("open_cost", 0.0015))
    if close_cost is None:
        close_cost = float(ex_kwargs_cfg.get("close_cost", 0.0015))
    if min_cost is None:
        min_cost = float(ex_kwargs_cfg.get("min_cost", 5.0))
    if save_report is None:
        save_report = int(pa_cfg.get("save_report", 1)) if "save_report" in pa_cfg else 1

    # 类型兜底
    freq = str(freq)
    topk = int(topk)
    n_drop = int(n_drop)
    account = float(account)
    save_report = int(save_report)
    if "start_time" in bt_cfg and "end_time" in bt_cfg:
        start_time = _as_timestamp(bt_cfg["start_time"])
        end_time_raw = _as_timestamp(bt_cfg["end_time"])
    else:
        seg = cfg["task"]["dataset"]["kwargs"]["segments"]["test"]
        start_time = _as_timestamp(seg[0])
        end_time_raw = _as_timestamp(seg[1])

    end_time = _safe_end_time_for_trade_calendar(end_time_raw, freq=str(freq))
    if pd.Timestamp(end_time) < pd.Timestamp(end_time_raw):
        print(
            f"[WARN] end_time 对齐/回退：raw_end={pd.Timestamp(end_time_raw).date()} -> end={pd.Timestamp(end_time).date()} "
            f"(原因：future calendar 不可用时需保证 end_index+1 存在)"
        )

    bench = _pick_benchmark(
        cfg=cfg,
        qlib_path=qlib_path,
        start_time=start_time,
        end_time=end_time,
        instruments_file=instruments_file if instruments_file.strip() != "" else None,
        benchmark_override=benchmark_override if benchmark_override.strip() != "" else None,
    )

    rows: List[Dict[str, Any]] = []
    for p in pred_files:
        seed, step = _parse_seed_step_from_pred_filename(p.name)
        pred = _read_predictions_csv(p)

        strategy = {
            "class": "TopkDropoutStrategy",
            "module_path": "qlib.contrib.strategy",
            "kwargs": {
                "signal": pred,
                "topk": int(topk),
                "n_drop": int(n_drop),
                "only_tradable": bool(only_tradable),
                "forbid_all_trade_at_limit": bool(forbid_all_trade_at_limit),
            },
        }
        executor = {
            "class": "SimulatorExecutor",
            "module_path": "qlib.backtest.executor",
            "kwargs": {
                "time_per_step": str(freq),
                "generate_portfolio_metrics": True,
            },
        }
        exchange_kwargs = {
            "deal_price": str(deal_price),
            "open_cost": float(open_cost),
            "close_cost": float(close_cost),
            "min_cost": float(min_cost),
            "limit_threshold": float(limit_threshold),
        }

        # 与 backtest_qlib.py 一致：边界异常降级重试
        cal = pd.DatetimeIndex(Cal.calendar(freq=str(freq), future=True))
        end_try = pd.Timestamp(end_time)
        last_err: Optional[Exception] = None
        for _ in range(3):
            try:
                port, _ind = qlib_backtest(
                    start_time=start_time,
                    end_time=end_try,
                    strategy=strategy,
                    executor=executor,
                    benchmark=bench,
                    account=float(account),
                    exchange_kwargs=exchange_kwargs,
                )
                break
            except IndexError as e:
                last_err = e
                i = int(cal.searchsorted(end_try, side="right") - 1)
                if i <= 0:
                    raise
                end_try = pd.Timestamp(cal[i - 1])
                print(f"[WARN] backtest IndexError，end_time 回退后重试：{end_try.date()}（seed={seed} step={step}）")
        else:
            assert last_err is not None
            raise last_err

        k = next(iter(port.keys()))
        report, _meta = port[k]
        if int(save_report) == 1:
            report.to_csv(out_root / f"report_seed_{seed}_step_{step}.csv", index=True)

        metrics = _analyze_report(report)
        metrics.update(
            {
                "Seed": int(seed),
                "Step": int(step),
                "pred_file": p.name,
                "start_time": str(pd.Timestamp(start_time).date()),
                "end_time": str(pd.Timestamp(end_try).date()),
            }
        )
        rows.append(metrics)

    df = pd.DataFrame(rows).sort_values(["Seed", "Step"]).reset_index(drop=True)
    df.to_csv(out_csv, index=False)
    print(f"[SUCCESS] backtest_result saved: {out_csv}")
    return str(out_csv)


if __name__ == "__main__":
    fire.Fire(main)

