"""
Sanity runner + data dumper for FuturesAlpha158 + DatasetH.

This script does NOT train a model; it checks that qlib can:
1) init with futures qlib bin
2) import FuturesAlpha158
3) prepare DatasetH segments successfully
4) dump train/valid/test prepared data to disk (optional)
"""

from __future__ import annotations

import argparse
import gc
import pickle
import sys
from datetime import datetime
from pathlib import Path

import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Sanity check for FuturesAlpha158 + DatasetH")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "workflow_config_futures_Alpha158.yaml"),
        help="Path to workflow config yaml",
    )
    parser.add_argument(
        "--save",
        type=int,
        default=1,
        help="1=save train/valid/test to disk; 0=only print shapes",
    )
    parser.add_argument(
        "--save_root",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "data" / "futures_results"),
        help="Root dir for dumped results",
    )
    parser.add_argument(
        "--folder_name",
        type=str,
        default="",
        help="Subfolder name under save_root (auto if empty)",
    )
    parser.add_argument(
        "--overwrite",
        type=int,
        default=0,
        help="1=overwrite existing output files; 0=skip if exists",
    )
    return parser.parse_args()


def _to_yyyymmdd(v) -> str:
    if hasattr(v, "strftime"):
        return v.strftime("%Y%m%d")
    s = str(v)
    return s.replace("-", "").replace("/", "")


def _auto_folder_name(config: dict) -> str:
    market = str(config.get("market", "all"))
    seg = config.get("task", {}).get("dataset", {}).get("kwargs", {}).get("segments", {})
    try:
        start_s = _to_yyyymmdd(seg["train"][0])
        end_s = _to_yyyymmdd(seg["test"][1])
    except Exception:
        start_s = "NA"
        end_s = "NA"
    run_date = datetime.now().strftime("%Y%m%d")
    return f"{market}_{run_date}_{start_s}_{end_s}"


def _dump_pickle(obj, path: Path, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and (not overwrite):
        print(f"[data_generator_futures] skip (exists): {path}")
        return
    with path.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[data_generator_futures] wrote: {path}")


def main():
    args = parse_args()
    # Ensure local qlib is importable (repo has qlib source under /home/idc2/notebook/qlib)
    sys.path.insert(0, "/home/idc2/notebook/qlib")
    # Ensure repo root is importable so `module_path: futures.*` works
    repo_root = str(Path(__file__).resolve().parents[2])
    sys.path.insert(0, repo_root)

    import qlib
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.utils import init_instance_by_config

    cfg_path = Path(args.config).expanduser().resolve()
    with cfg_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # YAML already contains region: cn
    qlib.init(**config["qlib_init"])

    dataset = init_instance_by_config(config["task"]["dataset"])
    dl_train = dataset.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
    dl_valid = dataset.prepare("valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
    dl_test = dataset.prepare("test", col_set=["feature", "label"], data_key=DataHandlerLP.DK_I)

    print("train", dl_train.shape, "valid", dl_valid.shape, "test", dl_test.shape)
    print("train head")
    print(dl_train.head())

    if int(args.save) != 1:
        return

    save_root = Path(args.save_root).expanduser().resolve()
    folder_name = args.folder_name.strip() or _auto_folder_name(config)
    save_dir = save_root / folder_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # Persist workflow config for reproducibility
    cfg_out = save_dir / cfg_path.name
    if (not cfg_out.exists()) or bool(int(args.overwrite)):
        with cfg_out.open("w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
        print(f"[data_generator_futures] wrote: {cfg_out}")

    universe = str(config.get("market", "all"))
    overwrite = bool(int(args.overwrite))

    _dump_pickle(dl_train, save_dir / f"{universe}_self_dl_train.pkl", overwrite=overwrite)
    del dl_train
    gc.collect()

    _dump_pickle(dl_valid, save_dir / f"{universe}_self_dl_valid.pkl", overwrite=overwrite)
    del dl_valid
    gc.collect()

    _dump_pickle(dl_test, save_dir / f"{universe}_self_dl_test.pkl", overwrite=overwrite)
    del dl_test
    gc.collect()

    print(f"[data_generator_futures] saved_dir={save_dir}")


if __name__ == "__main__":
    main()


