import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

import logging


logger = logging.getLogger("dump_futures_qlib_source")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


@dataclass(frozen=True)
class DumpResult:
    symbol: str
    success: bool
    message: str = ""


def _safe_int(x) -> int:
    try:
        return int(x)
    except Exception:
        return 0


def _aggregate_1m_to_1d_df(
    file_path: Union[str, Path],
    *,
    chunksize: int = 500_000,
    include_roll_features: bool = True,
    contract_multiplier: float = 1.0,
) -> pd.DataFrame:
    """
    Aggregate 1-minute futures klines into daily bars, grouped by trading_date.

    Notes
    -----
    - Assumes the source file is sorted by trading_date then datetime (as produced by `rq_f11_futures_klines_1m.py`).
    - Does NOT keep `dominant_id` as a column in the final output (string column would break qlib dump_bin).
    """
    file_path = Path(file_path)
    symbol = file_path.stem

    # Some files have different column orders, but column names are consistent.
    base_usecols = [
        "order_book_id",
        "datetime",
        "trading_date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "total_turnover",
        "open_interest",
    ]
    usecols = base_usecols + (["dominant_id"] if include_roll_features else [])

    daily_parts = []
    carry = None

    reader = pd.read_csv(
        file_path,
        usecols=lambda c: c in set(usecols),
        chunksize=chunksize,
        low_memory=False,
    )
    for chunk in reader:
        if chunk is None or chunk.empty:
            continue

        # unify basic columns
        if "total_turnover" in chunk.columns:
            chunk = chunk.rename(columns={"total_turnover": "amount"})

        # keep minimal columns for aggregation
        keep_cols = [
            "trading_date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "amount",
            "open_interest",
        ]
        if include_roll_features and "dominant_id" in chunk.columns:
            keep_cols.append("dominant_id")
        chunk = chunk.loc[:, [c for c in keep_cols if c in chunk.columns]]

        # attach carry-over incomplete day from previous chunk
        if carry is not None and not carry.empty:
            chunk = pd.concat([carry, chunk], ignore_index=True)
            carry = None

        if chunk.empty:
            continue

        # make sure trading_date is string for stable equality checks
        chunk["trading_date"] = chunk["trading_date"].astype(str)

        last_day = chunk["trading_date"].iloc[-1]
        last_mask = chunk["trading_date"] == last_day
        carry = chunk.loc[last_mask].copy()
        chunk_complete = chunk.loc[~last_mask]

        if chunk_complete.empty:
            continue

        agg_dict = {
            "open": "first",
            "close": "last",
            "high": "max",
            "low": "min",
            "volume": "sum",
            "amount": "sum",
            "open_interest": "last",
        }
        if include_roll_features and "dominant_id" in chunk_complete.columns:
            agg_dict["dominant_id"] = "last"

        daily = chunk_complete.groupby("trading_date", sort=True).agg(agg_dict).reset_index()

        if include_roll_features and "dominant_id" in chunk_complete.columns:
            # roll_count: number of intra-day dominant_id switches (minute-level)
            def _roll_cnt(s: pd.Series) -> int:
                if s.empty:
                    return 0
                # count changes within the day (exclude the first observation)
                return int(s.ne(s.shift(1)).sum() - 1)

            roll_cnt = (
                chunk_complete.groupby("trading_date", sort=True)["dominant_id"]
                .apply(_roll_cnt)
                .rename("roll_count")
                .reset_index()
            )
            daily = daily.merge(roll_cnt, on="trading_date", how="left")
        else:
            daily["roll_count"] = 0

        daily_parts.append(daily)

    # finalize last day
    if carry is not None and not carry.empty:
        if "total_turnover" in carry.columns:
            carry = carry.rename(columns={"total_turnover": "amount"})
        carry["trading_date"] = carry["trading_date"].astype(str)

        agg_dict = {
            "open": "first",
            "close": "last",
            "high": "max",
            "low": "min",
            "volume": "sum",
            "amount": "sum",
            "open_interest": "last",
        }
        if include_roll_features and "dominant_id" in carry.columns:
            agg_dict["dominant_id"] = "last"

        daily = carry.groupby("trading_date", sort=True).agg(agg_dict).reset_index()

        if include_roll_features and "dominant_id" in carry.columns:
            roll_cnt = int(carry["dominant_id"].ne(carry["dominant_id"].shift(1)).sum() - 1)
            daily["roll_count"] = max(0, roll_cnt)
        else:
            daily["roll_count"] = 0

        daily_parts.append(daily)

    if not daily_parts:
        return pd.DataFrame()

    daily_all = pd.concat(daily_parts, ignore_index=True)
    daily_all = daily_all.drop_duplicates(subset=["trading_date"], keep="last")
    daily_all = daily_all.sort_values("trading_date").reset_index(drop=True)

    # format output columns
    daily_all = daily_all.rename(columns={"trading_date": "tradedate"})
    daily_all["tradedate"] = pd.to_datetime(daily_all["tradedate"], errors="coerce")
    daily_all.insert(0, "symbol", symbol)

    # derived fields
    vol = pd.to_numeric(daily_all.get("volume", 0), errors="coerce")
    amt = pd.to_numeric(daily_all.get("amount", 0), errors="coerce")
    cm = float(contract_multiplier) if contract_multiplier and contract_multiplier > 0 else 1.0
    # futures turnover (amount) is typically `price * volume * contract_multiplier`
    daily_all["vwap"] = np.where(vol > 0, amt / (vol * cm + 1e-12), np.nan)
    daily_all["factor"] = 1.0
    daily_all["adjclose"] = daily_all["close"]

    if include_roll_features and "dominant_id" in daily_all.columns:
        # roll_flag: day-level dominant contract changes (based on end-of-day dominant_id)
        daily_all["roll_flag"] = (daily_all["dominant_id"] != daily_all["dominant_id"].shift(1)).astype(int)
        daily_all.loc[daily_all.index.min(), "roll_flag"] = 0
        # dominant_id stays internal only
        daily_all = daily_all.drop(columns=["dominant_id"])
    else:
        daily_all["roll_flag"] = 0

    # keep schema stable
    for col in ["open_interest", "roll_count", "roll_flag"]:
        if col not in daily_all.columns:
            daily_all[col] = 0

    # numeric coercions (avoid object dtype)
    num_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "vwap",
        "open_interest",
        "factor",
        "adjclose",
        "roll_count",
        "roll_flag",
    ]
    for c in num_cols:
        if c in daily_all.columns:
            daily_all[c] = pd.to_numeric(daily_all[c], errors="coerce")

    return daily_all


@lru_cache(maxsize=8)
def _load_contract_multiplier_map(futures_info_path: str) -> dict:
    p = Path(futures_info_path)
    if not p.exists():
        return {}
    try:
        df = pd.read_csv(p, usecols=["order_book_id", "contract_multiplier"], low_memory=False)
        df["order_book_id"] = df["order_book_id"].astype(str).str.upper()
        df["contract_multiplier"] = pd.to_numeric(df["contract_multiplier"], errors="coerce")
        df = df.dropna(subset=["order_book_id", "contract_multiplier"])
        df = df.drop_duplicates(subset=["order_book_id"], keep="last")
        return dict(zip(df["order_book_id"].tolist(), df["contract_multiplier"].tolist()))
    except Exception:
        return {}


def _get_contract_multiplier(symbol: str, futures_info_path: str) -> float:
    m = _load_contract_multiplier_map(str(futures_info_path))
    cm = m.get(str(symbol).upper(), 1.0)
    try:
        cm = float(cm)
    except Exception:
        cm = 1.0
    return cm if cm > 0 else 1.0


def _dump_single_symbol(
    file_path: Union[str, Path],
    *,
    out_raw_dir: Union[str, Path],
    out_backtest_dir: Union[str, Path],
    futures_info_path: Union[str, Path],
    chunksize: int,
    include_roll_features: bool,
    overwrite: bool,
) -> DumpResult:
    file_path = Path(file_path)
    symbol = file_path.stem
    try:
        out_raw_dir = Path(out_raw_dir)
        out_backtest_dir = Path(out_backtest_dir)
        out_raw_dir.mkdir(parents=True, exist_ok=True)
        out_backtest_dir.mkdir(parents=True, exist_ok=True)

        out_raw_file = out_raw_dir / f"{symbol}.csv"
        out_backtest_file = out_backtest_dir / f"{symbol}.csv"

        if (out_raw_file.exists() or out_backtest_file.exists()) and (not overwrite):
            return DumpResult(symbol=symbol, success=True, message="skip_exists")

        cm = _get_contract_multiplier(symbol, str(futures_info_path))
        daily_df = _aggregate_1m_to_1d_df(
            file_path,
            chunksize=chunksize,
            include_roll_features=include_roll_features,
            contract_multiplier=cm,
        )
        if daily_df is None or daily_df.empty:
            return DumpResult(symbol=symbol, success=False, message="empty_after_aggregate")

        # raw_data: keep more fields (include roll features)
        raw_cols = [
            "symbol",
            "tradedate",
            "open",
            "high",
            "low",
            "close",
            "vwap",
            "volume",
            "amount",
            "open_interest",
            "factor",
            "adjclose",
            "roll_count",
            "roll_flag",
        ]
        raw_df = daily_df.loc[:, [c for c in raw_cols if c in daily_df.columns]]

        # backtest_source: keep minimal fields (strictly numeric + required)
        bt_cols = [
            "symbol",
            "tradedate",
            "open",
            "high",
            "low",
            "close",
            "vwap",
            "volume",
            "amount",
            "open_interest",
            "factor",
            "adjclose",
        ]
        bt_df = daily_df.loc[:, [c for c in bt_cols if c in daily_df.columns]]
        bt_df = bt_df[~((bt_df["volume"] == 0) & (bt_df["amount"] == 0))]

        raw_df.to_csv(out_raw_file, index=False, date_format="%Y-%m-%d")
        bt_df.to_csv(out_backtest_file, index=False, date_format="%Y-%m-%d")

        return DumpResult(symbol=symbol, success=True, message="ok")
    except Exception as e:
        logger.exception("dump failed: %s - %s", symbol, e)
        return DumpResult(symbol=symbol, success=False, message=str(e))


def dump_futures_qlib_source(
    data_path: str = "/home/idc2/notebook/futures/data",
    in_1m_dir: str = "/home/idc2/notebook/futures/data/raw_data/futures_klines_1m",
    out_raw_dir: Optional[str] = None,
    out_backtest_dir: Optional[str] = None,
    symbols: Optional[Sequence[str]] = None,
    index_start: Optional[int] = None,
    index_end: Optional[int] = None,
    chunksize: int = 500_000,
    include_roll_features: bool = True,
    overwrite: bool = False,
    max_workers: Optional[int] = None,
):
    """
    Convert futures 1m raw data into daily qlib-source CSVs.

    Parameters
    ----------
    data_path : str
        Root futures data path.
    in_1m_dir : str
        Directory containing 1m CSV files.
    out_raw_dir : str
        Output directory for qlib source raw_data. Default: {data_path}/qlib_data/raw_data
    out_backtest_dir : str
        Output directory for qlib backtest source. Default: {data_path}/qlib_data/qlib_backtest_source
    symbols : list[str]
        Limit to specific symbols (file stems), e.g. ["RB88", "IF88"].
    index_start/index_end : int
        Slice the file list (useful for batching).
    """
    data_path = os.path.expanduser(data_path)
    in_1m_dir = Path(os.path.expanduser(in_1m_dir))
    futures_info_path = Path(f"{data_path}/raw_data/futures_info.csv")
    out_raw_dir = Path(out_raw_dir or f"{data_path}/qlib_data/raw_data")
    out_backtest_dir = Path(out_backtest_dir or f"{data_path}/qlib_data/qlib_backtest_source")

    files = sorted(in_1m_dir.glob("*.csv"))
    if symbols:
        sym_set = {str(s).upper() for s in symbols}
        files = [f for f in files if f.stem.upper() in sym_set]

    if index_start is not None or index_end is not None:
        files = files[_safe_int(index_start) : (_safe_int(index_end) if index_end is not None else None)]

    if not files:
        logger.warning(f"no input files found in: {in_1m_dir}")
        return

    if max_workers is None:
        cpu = os.cpu_count() or 2
        # I/O heavy; be conservative by default
        max_workers = max(1, min(cpu - 1, 8))

    total = len(files)
    ok = 0

    logger.info("dump futures qlib source start. files=%s, workers=%s, chunksize=%s", total, max_workers, chunksize)

    pbar = tqdm(total=total, desc="dump_futures_qlib_source", unit="file")
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        fut_map = {
            ex.submit(
                _dump_single_symbol,
                f,
                out_raw_dir=out_raw_dir,
                out_backtest_dir=out_backtest_dir,
                futures_info_path=futures_info_path,
                chunksize=chunksize,
                include_roll_features=include_roll_features,
                overwrite=overwrite,
            ): f
            for f in files
        }
        for fut in as_completed(fut_map):
            f = fut_map[fut]
            try:
                res: DumpResult = fut.result()
                if res.success:
                    ok += 1
                else:
                    logger.warning("failed: %s - %s", res.symbol, res.message)
            except Exception as e:
                logger.exception("unhandled exception: %s - %s", f, e)
            finally:
                pbar.update(1)
                pbar.set_postfix({"ok": ok, "fail": pbar.n - ok, "total": total})
    pbar.close()
    logger.info("dump futures qlib source done. ok=%s, fail=%s, total=%s", ok, total - ok, total)


if __name__ == "__main__":
    import argparse
    import ast

    parser = argparse.ArgumentParser(description="Dump futures 1m csv to daily qlib-source csv.")
    parser.add_argument("--data_path", type=str, default="/home/idc2/notebook/futures/data")
    parser.add_argument(
        "--in_1m_dir", type=str, default="/home/idc2/notebook/futures/data/raw_data/futures_klines_1m"
    )
    parser.add_argument("--out_raw_dir", type=str, default=None)
    parser.add_argument("--out_backtest_dir", type=str, default=None)
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help='Symbols to process. Supports python literal list like ["RB88","IF88"] or comma-separated string.',
    )
    parser.add_argument("--index_start", type=int, default=None)
    parser.add_argument("--index_end", type=int, default=None)
    parser.add_argument("--chunksize", type=int, default=500_000)
    parser.add_argument("--include_roll_features", type=int, default=1)
    parser.add_argument("--overwrite", type=int, default=0)
    parser.add_argument("--max_workers", type=int, default=None)
    args = parser.parse_args()

    symbols = None
    if args.symbols:
        try:
            parsed = ast.literal_eval(args.symbols)
            if isinstance(parsed, (list, tuple, set)):
                symbols = list(parsed)
            elif isinstance(parsed, str):
                symbols = [parsed]
        except Exception:
            symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]

    dump_futures_qlib_source(
        data_path=args.data_path,
        in_1m_dir=args.in_1m_dir,
        out_raw_dir=args.out_raw_dir,
        out_backtest_dir=args.out_backtest_dir,
        symbols=symbols,
        index_start=args.index_start,
        index_end=args.index_end,
        chunksize=args.chunksize,
        include_roll_features=bool(args.include_roll_features),
        overwrite=bool(args.overwrite),
        max_workers=args.max_workers,
    )


