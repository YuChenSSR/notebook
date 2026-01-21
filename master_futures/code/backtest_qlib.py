"""
backtest_qlib.py

期货 f88：读取 MASTER 预测文件（master_predictions_*.csv），用 Qlib 做仿真回测（TopkDropoutStrategy + SimulatorExecutor）。

设计目标：
- 作为 master_futures_oneclick.sh 的 Step C：训练→预测后直接回测并落盘结果
- 默认不依赖“股票基准”（CSI300 等）；benchmark 将从 instruments/f88.txt 自动构造等权基准（或由用户显式指定）
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
REPO_ROOT = str(_THIS_FILE.parents[2])  # .../notebook
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
    # yaml.safe_load 对 yyyy-mm-dd 通常会解析成 date/datetime；统一转 Timestamp
    return pd.Timestamp(x)


def _read_predictions_csv(path: Union[str, Path]) -> pd.Series:
    """
    读取预测文件，返回 MultiIndex(datetime, instrument) 的 score Series。
    期望列：datetime, instrument, score
    """
    path = Path(path).expanduser().resolve()
    df = pd.read_csv(path)
    if not {"datetime", "instrument", "score"}.issubset(df.columns):
        raise ValueError(f"预测文件缺少列（需要 datetime/instrument/score）：{path} cols={list(df.columns)}")
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["instrument"] = df["instrument"].astype(str)
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df.dropna(subset=["score"])
    s = df.set_index(["datetime", "instrument"])["score"].sort_index()
    return s


def _parse_seed_from_pred_filename(filename: str) -> Optional[int]:
    """
    期货版 Master/main.py 输出文件名：
      master_predictions_backday_{backday}_{universe_tag}_{seed}.csv
    universe_tag 可能包含 '_'（例如 f88_rank），因此只从末尾解析 seed。
    """
    stem = filename.rsplit(".", 1)[0]
    parts = stem.split("_")
    if len(parts) < 2:
        return None
    seed_str = parts[-1]
    return int(seed_str) if seed_str.isdigit() else None


def _load_f88_codes_from_instruments_file(
    instruments_path: Union[str, Path],
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
) -> List[str]:
    """
    从 instruments/f88.txt 读取合约列表，并按回测区间过滤（仅保留与区间有交集的合约）。

    文件格式（tab 分隔）：
      SYMBOL<TAB>start_date<TAB>end_date
    """
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
            # 与 [start_time, end_time] 无交集则跳过
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
    """
    决定 benchmark：
    1) 若 benchmark_override 非空：直接用（支持单合约代码）
    2) 若 cfg.port_analysis_config.backtest.benchmark 非空：直接用（支持 str 或 list）
    3) 否则：从 instruments/f88.txt 构造等权基准（list[str]）
    """
    if benchmark_override is not None and str(benchmark_override).strip() != "":
        return str(benchmark_override).strip()

    bt_cfg = (cfg.get("port_analysis_config", {}) or {}).get("backtest", {}) or {}
    bench = bt_cfg.get("benchmark", None)
    if bench is not None:
        # 允许 list/str；空字符串按未提供处理
        if isinstance(bench, str) and bench.strip() != "":
            return bench.strip()
        if isinstance(bench, list) and len(bench) > 0:
            return [str(x) for x in bench]

    if instruments_file is None or str(instruments_file).strip() == "":
        instruments_file = os.path.join(os.path.expanduser(qlib_path), "instruments", "f88.txt")
    return _load_f88_codes_from_instruments_file(instruments_file, start_time=start_time, end_time=end_time)


def _analyze_report(report: pd.DataFrame) -> Dict[str, float]:
    """
    从 Qlib report 计算常用指标。
    - return: 组合毛收益
    - cost  : 交易费用
    - bench : 基准收益（若提供 benchmark）
    """
    # 净收益
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


def _safe_end_time_for_trade_calendar(end_time: pd.Timestamp, freq: str = "day") -> pd.Timestamp:
    """
    Qlib 的 TradeCalendarManager.get_step_time 会访问 `calendar_index + 1`（用于闭区间右端点）。
    在某些自建数据集里，`Cal.calendar(freq, future=True)` 并不会比“当前日历”多出一个未来交易日，
    这会导致当 end_time 恰好落在日历最后一天时，出现 IndexError（越界访问 calendar_index+1）。

    这里做一个稳健兜底：
    - 将 end_time 对齐到 <=end_time 的最后一个交易日
    - 若该交易日已是日历最后一天，则再往前退 1 个交易日
    """
    cal = pd.DatetimeIndex(Cal.calendar(freq=freq, future=True))
    if len(cal) < 3:
        raise RuntimeError(f"交易日历长度异常（len={len(cal)}），无法回测：freq={freq}")

    end_time = pd.Timestamp(end_time)
    # 对齐到 <= end_time 的交易日索引
    i = int(cal.searchsorted(end_time, side="right") - 1)
    if i < 0:
        raise ValueError(f"end_time 早于交易日历起点：end_time={end_time.date()} earliest={cal[0].date()}")

    # 确保 i+1 有效（get_step_time 需要）
    if i >= len(cal) - 1:
        i = len(cal) - 2
    return pd.Timestamp(cal[i])


def _resolve_pred_dir(pred_dir: str, experimental_data_path: str) -> Path:
    """
    解析预测目录：
    - 若用户显式传入 pred_dir：
        - 绝对路径：直接使用
        - 相对路径：相对 experimental_data_path
    - 否则默认优先使用 Backtest_Results/predictions（更贴近股票版流程），不存在则回退到 Master_results
    """
    exp_dir = Path(experimental_data_path).expanduser().resolve()
    if pred_dir is not None and str(pred_dir).strip() != "":
        p = Path(str(pred_dir)).expanduser()
        if not p.is_absolute():
            p = exp_dir / p
        return p.resolve()

    cand = exp_dir / "Backtest_Results" / "predictions"
    if cand.exists():
        return cand
    return exp_dir / "Master_results"


def main(
    market_name: str = "f88",
    folder_name: str = "f88_20260113_20100104_20251212",
    data_path: str = "/home/idc2/notebook/master_futures/data",
    qlib_path: str = "/home/idc2/notebook/futures/data/qlib_bin/cn_data_backtest",
    instruments_file: str = "",
    benchmark_override: str = "",
    topk: int = 30,
    n_drop: int = 3,
    account: float = 100_000_000,
    freq: str = "day",
    deal_price: str = "close",
    open_cost: float = 0.0015,
    close_cost: float = 0.0015,
    min_cost: float = 5.0,
    limit_threshold: float = 0.095,
    save_report: int = 1,
    pred_dir: str = "",
) -> str:
    """
    读取预测文件并批量回测，输出 backtest_result.csv。

    返回：backtest_result.csv 的路径
    """
    data_path = os.path.abspath(os.path.expanduser(data_path))
    experimental_data_path = os.path.join(data_path, "master_results", folder_name)

    cfg_path = os.path.join(experimental_data_path, f"workflow_config_master_Alpha158_{market_name}.yaml")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"找不到实验 config：{cfg_path}")
    cfg = _load_yaml(cfg_path)

    # Qlib init（回测用 provider_uri）
    qlib.init(provider_uri=os.path.expanduser(qlib_path), region=REG_CN)

    # 回测区间：优先用 port_analysis_config.backtest；否则用 dataset 的 test segment
    bt_cfg = (cfg.get("port_analysis_config", {}) or {}).get("backtest", {}) or {}
    if "start_time" in bt_cfg and "end_time" in bt_cfg:
        start_time = _as_timestamp(bt_cfg["start_time"])
        end_time_raw = _as_timestamp(bt_cfg["end_time"])
    else:
        seg = cfg["task"]["dataset"]["kwargs"]["segments"]["test"]
        start_time = _as_timestamp(seg[0])
        end_time_raw = _as_timestamp(seg[1])

    # 关键兜底：避免 end_time 落在日历最后一天导致 get_step_time 越界
    end_time = _safe_end_time_for_trade_calendar(end_time_raw, freq=str(freq))
    if pd.Timestamp(end_time) < pd.Timestamp(end_time_raw):
        print(
            f"[WARN] end_time 对齐/回退：raw_end={pd.Timestamp(end_time_raw).date()} -> end={pd.Timestamp(end_time).date()} "
            f"(原因：future calendar 不可用时需保证 end_index+1 存在)"
        )

    # benchmark：避免 Qlib 默认 CSI300
    bench = _pick_benchmark(
        cfg=cfg,
        qlib_path=qlib_path,
        start_time=start_time,
        end_time=end_time,
        instruments_file=instruments_file if instruments_file.strip() != "" else None,
        benchmark_override=benchmark_override if benchmark_override.strip() != "" else None,
    )

    # 预测文件目录：默认优先 Backtest_Results/predictions（更贴近股票版流程），不存在则回退到 Master_results
    pred_dir_path = _resolve_pred_dir(pred_dir=pred_dir, experimental_data_path=experimental_data_path)
    if not pred_dir_path.exists():
        raise FileNotFoundError(f"找不到预测目录：{pred_dir_path}")

    pred_files = sorted([p for p in pred_dir_path.glob("master_predictions_backday_*.csv") if p.is_file()])
    if len(pred_files) == 0:
        raise FileNotFoundError(f"未找到预测文件：{pred_dir_path}/master_predictions_backday_*.csv")

    # 输出目录
    out_root = Path(experimental_data_path) / "Backtest_Results" / "results"
    out_root.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for p in pred_files:
        seed = _parse_seed_from_pred_filename(p.name)
        if seed is None:
            # 避免误吞其它文件
            continue

        pred = _read_predictions_csv(p)

        strategy = {
            "class": "TopkDropoutStrategy",
            "module_path": "qlib.contrib.strategy",
            "kwargs": {
                "signal": pred,
                "topk": int(topk),
                "n_drop": int(n_drop),
                "only_tradable": False,
                "forbid_all_trade_at_limit": False,
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

        # 某些数据集的 future calendar 不完整时，Qlib 仍可能因边界触发 IndexError；这里做一次降级重试
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
                # 往前退 1 个交易日再试
                i = int(cal.searchsorted(end_try, side="right") - 1)
                if i <= 0:
                    raise
                end_try = pd.Timestamp(cal[i - 1])
                print(f"[WARN] backtest IndexError，end_time 回退后重试：{end_try.date()}（seed={seed}）")
        else:
            assert last_err is not None
            raise last_err

        # 频率 key 通常为 "1day"
        k = next(iter(port.keys()))
        report, _meta = port[k]

        if int(save_report) == 1:
            report_path = out_root / f"report_seed_{seed}.csv"
            report.to_csv(report_path, index=True)

        metrics = _analyze_report(report)
        metrics.update(
            {
                "seed": int(seed),
                "pred_file": p.name,
                "start_time": str(pd.Timestamp(start_time).date()),
                "end_time": str(pd.Timestamp(end_try).date()),
            }
        )
        rows.append(metrics)

    if len(rows) == 0:
        raise RuntimeError("未解析到任何可回测的预测文件（文件名末尾 seed 解析失败？）")

    df = pd.DataFrame(rows).sort_values(["seed"]).reset_index(drop=True)
    out_csv = out_root / "backtest_result.csv"
    df.to_csv(out_csv, index=False)
    print(f"[SUCCESS] backtest_result saved: {out_csv}")
    return str(out_csv)


if __name__ == "__main__":
    fire.Fire(main)

