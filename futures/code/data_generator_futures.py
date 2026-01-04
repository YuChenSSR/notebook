"""
Small sanity runner for FuturesAlpha158 + DatasetH.

This script does NOT train a model; it just checks that qlib can:
1) init with futures qlib bin
2) import FuturesAlpha158
3) prepare DatasetH segments successfully
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml


def main():
    # Ensure local qlib is importable (repo has qlib source under /home/idc2/notebook/qlib)
    sys.path.insert(0, "/home/idc2/notebook/qlib")
    # Ensure repo root is importable so `module_path: futures.*` works
    repo_root = str(Path(__file__).resolve().parents[2])
    sys.path.insert(0, repo_root)

    import qlib
    from qlib.utils import init_instance_by_config

    cfg_path = Path(__file__).resolve().parents[1] / "workflow_config_futures_Alpha158.yaml"
    with cfg_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # YAML already contains region: cn
    qlib.init(**config["qlib_init"])

    dataset = init_instance_by_config(config["task"]["dataset"])
    dl_train = dataset.prepare("train", col_set=["feature", "label"])
    dl_valid = dataset.prepare("valid", col_set=["feature", "label"])
    dl_test = dataset.prepare("test", col_set=["feature", "label"])

    print("train", dl_train.shape, "valid", dl_valid.shape, "test", dl_test.shape)
    print("train head")
    print(dl_train.head())


if __name__ == "__main__":
    main()


