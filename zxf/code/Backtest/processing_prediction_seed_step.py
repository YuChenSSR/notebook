import os
import sys
import pickle
import numpy as np
import time
import fire
import pandas as pd
import yaml
import torch
from pathlib import Path

PROJ_DIRNAME = '/home/idc2/notebook/zxf/code'
sys.path.insert(0, PROJ_DIRNAME)
from Master.master import MASTERModel


def _parse_seed_step_from_filename(filename: str) -> tuple[int, int]:
    """
    从文件名末尾提取 seed 与 step，避免 universe/前缀包含 '_' 时 split 固定长度导致报错。

    例如：
    - csi800_backday_8_self_exp_17_0.pkl
    - csi800_rank_backday_8_self_exp_17_0.pkl
    """
    stem = filename.rsplit(".", 1)[0]
    parts = stem.split("_")
    if len(parts) < 2:
        raise ValueError(f"Invalid filename: {filename}")
    seed_str, step_str = parts[-2], parts[-1]
    if not (seed_str.isdigit() and step_str.isdigit()):
        raise ValueError(f"Cannot parse seed/step from filename: {filename}")
    return int(seed_str), int(step_str)


### 1. 加工特征数据
def processing_predictions(workflow_path,Data_path):

    ### 1. 读取workflow & 设置训练参数【需与实验一致】
    with open(workflow_path, 'r') as f:
        config = yaml.safe_load(f)
    universe = config["market"]  # ['csi300','csi800']
    backday = config['task']['dataset']['kwargs']['step_len']

    d_feat = config["task"]["model"]["kwargs"]["d_feat"]                                        # d_feat = 175    # 158
    d_model = config["task"]["model"]["kwargs"]["d_model"]                                      # d_model = 256
    t_nhead = config["task"]["model"]["kwargs"]["t_nhead"]                                      # t_nhead = 4
    s_nhead = config["task"]["model"]["kwargs"]["s_nhead"]                                      # s_nhead = 2
    dropout = config["task"]["model"]["kwargs"]["dropout"]                                      # dropout = 0.2  # 0.5
    gate_input_start_index = config["task"]["model"]["kwargs"]["gate_input_start_index"]        # gate_input_start_index = 175    #158
    gate_input_end_index = config["task"]["model"]["kwargs"]["gate_input_end_index"]            # gate_input_end_index = 237      # 221
    n_epoch = config["task"]["model"]["kwargs"]["n_epochs"]                                      # n_epoch = 100
    lr = config["task"]["model"]["kwargs"]["lr"]                                                # lr = 1e-5
    GPU = config["task"]["model"]["kwargs"]["GPU"]                                              # GPU = 0
    train_stop_loss_thred = config["task"]["model"]["kwargs"]["train_stop_loss_thred"]          # train_stop_loss_thred = 0.92
    beta = config["task"]["model"]["kwargs"]["beta"]                                            # csi300:5;csi500:3;csi800:2;else:2
    benchmark =  config["benchmark"]


    ### 2. 读取测试数据集
    with open(f'{Data_path}/{universe}_self_dl_test.pkl', 'rb') as f:
        dl_test = pickle.load(f)


    ### 2. 读取因子
    save_path = f'{Data_path}/Backtest_Results/predictions'
    os.makedirs(save_path, exist_ok=True)

    factor_path = f'{Data_path}/Master_results'
    factor_folder_path = Path(factor_path)
    factor_folder_path = [file.name for file in factor_folder_path.glob("*self_exp*.pkl")]

    test_process_info = pd.DataFrame([])
    for filename in factor_folder_path:
        seed, step = _parse_seed_step_from_filename(filename)
        print(f"Filename:{filename} / Seed:{seed} / Step:{step}")

        # 加载和转化因子
        factor_file_path = Path(factor_path) / filename
        old_state_dict = torch.load(factor_file_path)

        new_keys = ["encoder.0.weight", "encoder.0.bias", "encoder.1.pe", "encoder.2.qtrans.weight", "encoder.2.ktrans.weight", "encoder.2.vtrans.weight", "encoder.2.norm1.weight", "encoder.2.norm1.bias", "encoder.2.norm2.weight", "encoder.2.norm2.bias", "encoder.2.ffn.0.weight", "encoder.2.ffn.0.bias", "encoder.2.ffn.3.weight", "encoder.2.ffn.3.bias", "encoder.3.qtrans.weight", "encoder.3.ktrans.weight", "encoder.3.vtrans.weight", "encoder.3.norm1.weight", "encoder.3.norm1.bias", "encoder.3.norm2.weight", "encoder.3.norm2.bias", "encoder.3.ffn.0.weight", "encoder.3.ffn.0.bias", "encoder.3.ffn.3.weight", "encoder.3.ffn.3.bias", "encoder.4.trans.weight", "decoder.weight", "decoder.bias"]
        old_keys = ["layers.0.weight", "layers.0.bias", "layers.1.pe", "layers.2.qtrans.weight", "layers.2.ktrans.weight", "layers.2.vtrans.weight", "layers.2.norm1.weight", "layers.2.norm1.bias", "layers.2.norm2.weight", "layers.2.norm2.bias", "layers.2.ffn.0.weight", "layers.2.ffn.0.bias", "layers.2.ffn.3.weight", "layers.2.ffn.3.bias", "layers.3.qtrans.weight", "layers.3.ktrans.weight", "layers.3.vtrans.weight", "layers.3.norm1.weight", "layers.3.norm1.bias", "layers.3.norm2.weight", "layers.3.norm2.bias", "layers.3.ffn.0.weight", "layers.3.ffn.0.bias", "layers.3.ffn.3.weight", "layers.3.ffn.3.bias", "layers.4.trans.weight", "layers.5.weight", "layers.5.bias"]
        key_tsfm = {old_key: new_key for old_key, new_key in zip(old_keys, new_keys)}

        new_state_dict = {key_tsfm.get(k, k): v for k, v in old_state_dict.items()}

        # 加载实验环境生成pred
        model = MASTERModel(
                d_feat = d_feat, d_model = d_model, t_nhead = t_nhead, s_nhead = s_nhead, T_dropout_rate=dropout, S_dropout_rate=dropout,
                beta=beta, gate_input_end_index=gate_input_end_index, gate_input_start_index=gate_input_start_index,
                n_epochs=n_epoch, lr = lr, GPU = GPU, seed = seed, train_stop_loss_thred = train_stop_loss_thred,
                save_path=save_path, save_prefix=universe
            )
        model.model.load_state_dict(new_state_dict)
        model.fitted = 1

        predictions, metrics = model.predict(dl_test)

        df = {
            'Seed': seed,
            'Step': step,
            'Test_IC': metrics['IC'],
            'Test_ICIR': metrics['ICIR'],
            'Test_RIC': metrics['RIC'],
            'Test_RICIR': metrics['RICIR']
        }
        test_process_info = pd.concat([test_process_info, pd.DataFrame([df])], ignore_index=True)

        pred_frame = predictions.to_frame()
        pred_frame.columns = ['score']
        pred_frame.reset_index(inplace=True)
        pred_frame.to_csv(f'{save_path}/master_predictions_backday_{backday}_{universe}_{seed}_{step}.csv', index=False, date_format='%Y-%m-%d')

    test_process_info.to_csv(f'{save_path}/test_metrics_results.csv', index=False)

def main(
        # project_dir: str = "/c/Data_processing",
        # project_dir: str = "c:/Quant",
        data_path: str = "/home/idc2/notebook/zxf/data",
        market_name: str = "csi800",
        folder_name: str = "csi800_20251105_20150101_20251103",
):

    workflow_path = f"{data_path}/master_results/{folder_name}/workflow_config_master_Alpha158_{market_name}.yaml"
    Data_path = f'{data_path}/master_results/{folder_name}'

    processing_predictions(workflow_path=workflow_path, Data_path=Data_path)


if __name__ == "__main__":
  fire.Fire(main)