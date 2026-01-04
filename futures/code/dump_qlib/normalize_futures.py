import ast
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from tqdm import tqdm


logger = logging.getLogger("normalize_futures")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


@dataclass(frozen=True)
class NormResult:
    symbol: str
    success: bool
    message: str = ""


@lru_cache(maxsize=8)
def _load_contract_multiplier_map(futures_info_path: str) -> Dict[str, float]:
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


def _read_csv(csv_path: Union[str, Path], usecols: Optional[List[str]] = None) -> pd.DataFrame:
    return pd.read_csv(csv_path, usecols=usecols, low_memory=False)


def build_trade_calendar(
    raw_dir: Union[str, Path],
    *,
    calendar_file: Union[str, Path],
    overwrite: bool = False,
    limit_files: Optional[int] = None,
) -> pd.DataFrame:
    """
    Build a global futures trading calendar from the union of `tradedate` in raw qlib-source CSVs.
    """
    raw_dir = Path(raw_dir)
    calendar_file = Path(calendar_file)
    calendar_file.parent.mkdir(parents=True, exist_ok=True)

    if calendar_file.exists() and not overwrite:
        cal = pd.read_csv(calendar_file)
        cal["tradedate"] = pd.to_datetime(cal["tradedate"], errors="coerce")
        cal = cal.dropna(subset=["tradedate"]).drop_duplicates(subset=["tradedate"]).sort_values("tradedate")
        return cal.reset_index(drop=True)

    files = sorted(raw_dir.glob("*.csv"))
    if limit_files is not None:
        files = files[: int(limit_files)]

    if not files:
        raise FileNotFoundError(f"no raw csv files found under: {raw_dir}")

    dates = set()
    for f in tqdm(files, desc="build_trade_calendar", unit="file"):
        try:
            s = _read_csv(f, usecols=["tradedate"])["tradedate"]
            s = pd.to_datetime(s, errors="coerce")
            dates.update(set(s.dropna().dt.normalize().tolist()))
        except Exception:
            continue

    cal = pd.DataFrame({"tradedate": sorted(dates)})
    cal.to_csv(calendar_file, index=False, date_format="%Y-%m-%d")
    return cal


def _compute_base_close(close_s: pd.Series, window: int = 30) -> float:
    close_s = pd.to_numeric(close_s, errors="coerce")
    close_s = close_s.replace([np.inf, -np.inf], np.nan).dropna()
    close_s = close_s[close_s > 0]
    if close_s.empty:
        return 1.0
    if len(close_s) >= window:
        return float(close_s.iloc[:window].median())
    return float(close_s.iloc[0])


def _normalize_one(
    file_path: Union[str, Path],
    *,
    out_dir: Union[str, Path],
    calendar_df: pd.DataFrame,
    futures_info_path: Union[str, Path],
    overwrite: bool,
    base_window: int,
) -> NormResult:
    file_path = Path(file_path)
    symbol = file_path.stem
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{symbol}.csv"

    try:
        if out_file.exists() and not overwrite:
            return NormResult(symbol=symbol, success=True, message="skip_exists")

        df = _read_csv(file_path)
        if df is None or df.empty:
            return NormResult(symbol=symbol, success=False, message="empty_input")

        if "symbol" not in df.columns:
            df.insert(0, "symbol", symbol)
        df["symbol"] = symbol

        df["tradedate"] = pd.to_datetime(df["tradedate"], errors="coerce")
        df = df.dropna(subset=["tradedate"]).sort_values("tradedate").reset_index(drop=True)
        if df.empty:
            return NormResult(symbol=symbol, success=False, message="empty_after_parse")

        # keep raw adjclose for label
        df["adjclose"] = df["close"]

        base_close = _compute_base_close(df["close"], window=base_window)
        cm = _get_contract_multiplier(symbol, str(futures_info_path))

        # normalize price-like columns
        price_cols = ["open", "high", "low", "close", "vwap"]
        for c in price_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce") / (base_close + 1e-12)

        # normalize volume into notional-ish scale (contracts -> money proxy)
        if "volume" in df.columns:
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce") * base_close * cm

        # calendar align within [min, max]
        cal = calendar_df[
            (calendar_df["tradedate"] >= df["tradedate"].min()) & (calendar_df["tradedate"] <= df["tradedate"].max())
        ].copy()
        df = pd.merge(cal, df, on="tradedate", how="outer")
        df["symbol"] = symbol
        df = df.sort_values("tradedate").reset_index(drop=True)

        # columns where 0 is a valid value and should not be treated as missing
        zero_ok_cols = {"roll_count", "roll_flag"}
        fill_one_cols = {"factor"}

        # mark invalid non-positive values as NaN for continuous features (but keep roll_* intact)
        check_cols = [
            "open",
            "high",
            "low",
            "close",
            "vwap",
            "volume",
            "amount",
            "open_interest",
            "adjclose",
        ]
        for c in check_cols:
            if c in df.columns and c not in zero_ok_cols:
                v = pd.to_numeric(df[c], errors="coerce")
                df[c] = np.where(v <= 0, np.nan, v)

        # fill factor and roll features first
        for c in fill_one_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(1.0)
        for c in zero_ok_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

        # fill all numeric columns (except date/symbol) with ffill/bfill/1e-12
        for c in df.columns:
            if c in {"tradedate", "symbol"}:
                continue
            if c in zero_ok_cols or c in fill_one_cols:
                continue
            s = pd.to_numeric(df[c], errors="coerce")
            df[c] = s.ffill().bfill().fillna(1e-12)

        # stable column ordering (similar to qlib expectations)
        preferred = [
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
        cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
        df = df.loc[:, cols]

        df.to_csv(out_file, index=False, date_format="%Y-%m-%d")
        return NormResult(symbol=symbol, success=True, message="ok")
    except Exception as e:
        logger.exception("normalize failed: %s - %s", symbol, e)
        return NormResult(symbol=symbol, success=False, message=str(e))


def normalize_futures(
    data_path: str = "/home/idc2/notebook/futures/data",
    raw_dir: Optional[str] = None,
    out_dir: Optional[str] = None,
    calendar_file: Optional[str] = None,
    overwrite_calendar: int = 0,
    overwrite: int = 0,
    max_workers: Optional[int] = None,
    symbols: Optional[Sequence[str]] = None,
    index_start: Optional[int] = None,
    index_end: Optional[int] = None,
    base_window: int = 30,
):
    data_path = os.path.expanduser(data_path)
    raw_dir = Path(raw_dir or f"{data_path}/qlib_data/raw_data")
    out_dir = Path(out_dir or f"{data_path}/qlib_data/qlib_train_source")
    futures_info_path = Path(f"{data_path}/raw_data/futures_info.csv")
    calendar_file = Path(calendar_file or f"{data_path}/qlib_data/trade_calendar.csv")

    files = sorted(raw_dir.glob("*.csv"))
    if symbols:
        sym_set = {str(s).upper() for s in symbols}
        files = [f for f in files if f.stem.upper() in sym_set]
    if index_start is not None or index_end is not None:
        s = int(index_start or 0)
        e = int(index_end) if index_end is not None else None
        files = files[s:e]

    if not files:
        raise FileNotFoundError(f"no raw csv files to normalize under: {raw_dir}")

    calendar_df = build_trade_calendar(
        raw_dir,
        calendar_file=calendar_file,
        overwrite=bool(overwrite_calendar),
    )

    if max_workers is None:
        cpu = os.cpu_count() or 2
        max_workers = max(1, min(cpu - 1, 8))

    total = len(files)
    ok = 0
    logger.info("normalize futures start. files=%s, workers=%s", total, max_workers)

    pbar = tqdm(total=total, desc="normalize_futures", unit="file")
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        fut_map = {
            ex.submit(
                _normalize_one,
                f,
                out_dir=out_dir,
                calendar_df=calendar_df,
                futures_info_path=futures_info_path,
                overwrite=bool(overwrite),
                base_window=base_window,
            ): f
            for f in files
        }
        for fut in as_completed(fut_map):
            f = fut_map[fut]
            try:
                res: NormResult = fut.result()
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
    logger.info("normalize futures done. ok=%s, fail=%s, total=%s", ok, total - ok, total)


def _parse_symbols(s: Optional[str]) -> Optional[List[str]]:
    if not s:
        return None
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, (list, tuple, set)):
            return [str(x) for x in parsed]
        if isinstance(parsed, str):
            return [parsed]
    except Exception:
        pass
    return [x.strip() for x in s.split(",") if x.strip()]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Normalize futures daily qlib-source data (scale + calendar align).")
    parser.add_argument("--data_path", type=str, default="/home/idc2/notebook/futures/data")
    parser.add_argument("--raw_dir", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--calendar_file", type=str, default=None)
    parser.add_argument("--overwrite_calendar", type=int, default=0)
    parser.add_argument("--overwrite", type=int, default=0)
    parser.add_argument("--max_workers", type=int, default=None)
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help='Symbols to process. Supports python literal list like ["RB88","IF88"] or comma-separated string.',
    )
    parser.add_argument("--index_start", type=int, default=None)
    parser.add_argument("--index_end", type=int, default=None)
    parser.add_argument("--base_window", type=int, default=30)
    args = parser.parse_args()

    normalize_futures(
        data_path=args.data_path,
        raw_dir=args.raw_dir,
        out_dir=args.out_dir,
        calendar_file=args.calendar_file,
        overwrite_calendar=args.overwrite_calendar,
        overwrite=args.overwrite,
        max_workers=args.max_workers,
        symbols=_parse_symbols(args.symbols),
        index_start=args.index_start,
        index_end=args.index_end,
        base_window=args.base_window,
    )


