import sys
import os
import yaml
import fire
import pickle
from pathlib import Path



# 将本地 qlib 源码目录设置为最高优先级（仓库内自带）
QLIB_DIRNAME = "/home/idc2/notebook/qlib"
if QLIB_DIRNAME not in sys.path:
    sys.path.insert(0, QLIB_DIRNAME)

# 确保 repo root 可 import（用于 module_path: master_futures.*）
_THIS_FILE = Path(__file__).resolve()
REPO_ROOT = str(_THIS_FILE.parents[2])  # .../notebook
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config
from qlib.data.dataset.handler import DataHandlerLP

def data_generator(
        market_name: str = "f88",
        qlib_path: str = "/home/idc2/notebook/futures/data/qlib_bin/cn_data_train",
        data_path: str = "/home/idc2/notebook/master_futures/data",
        folder_name: str = "f88_20260101_20100104_20251212",
        config_path: str | None = None,
        overwrite_exp_config: bool = True,
):

    qlib.init(provider_uri=qlib_path, region=REG_CN)
    if config_path is None or str(config_path).strip() == "":
        config_path = f"{data_path}/workflow_config_master_Alpha158_{market_name}.yaml"
    with open(str(config_path), 'r') as f:
        config = yaml.safe_load(f)

    start_date = config["task"]["dataset"]["kwargs"]["segments"]["train"][0].strftime("%Y%m%d")
    end_date = config["task"]["dataset"]["kwargs"]["segments"]["test"][1].strftime("%Y%m%d")


    save_path = f"{data_path}/master_results/{folder_name}"
    os.makedirs(save_path, exist_ok=True)

    h_conf = config["task"]["dataset"]["kwargs"]["handler"]
    h_path = Path(f'{save_path}/' + f'handler_{start_date}_{end_date}.pkl')

    if not h_path.exists():
        h = init_instance_by_config(h_conf)
        h.to_pickle(h_path, dump_all=True)
        del h

    config["task"]["dataset"]["kwargs"]["handler"] = f"file://{h_path}"
    dataset = init_instance_by_config(config['task']["dataset"])

    # 将本次使用的 workflow_config 写入实验目录，保证可追溯/可复现
    exp_cfg_path = Path(save_path) / f"workflow_config_master_Alpha158_{market_name}.yaml"
    if overwrite_exp_config or (not exp_cfg_path.exists()):
        with exp_cfg_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)

    dl_train = dataset.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
    with open(f'{save_path}/{market_name}_self_dl_train.pkl', 'wb') as file: pickle.dump(dl_train, file)
    del dl_train
    
    dl_valid = dataset.prepare("valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
    with open(f'{save_path}/{market_name}_self_dl_valid.pkl', 'wb') as file: pickle.dump(dl_valid, file)
    del dl_valid
    
    dl_test = dataset.prepare("test", col_set=["feature", "label"], data_key=DataHandlerLP.DK_I)
    with open(f'{save_path}/{market_name}_self_dl_test.pkl', 'wb') as file: pickle.dump(dl_test, file)
    del dl_test

if __name__ == "__main__":
    fire.Fire(data_generator)