import pandas as pd
import pickle
import yaml
import torch
import fire
import sys
from qlib import init
from qlib.constant import REG_CN

sys.path.append('/home/idc2/notebook/zxf/code')
from Master.master import MASTERModel


def main(
        data_path: str = "/home/idc2/notebook/zxf/data",
        qlib_path: str = '/home/idc2/notebook/qlib_bin/cn_data_train',
        market_name: str = "csi800",
        folder_name: str = "csi800c_20251209_20200101_20251208",
        seed: int = 135,
        step: int = 61,
        d_model: int = 128,
):
    expt_path = f"{data_path}/beta_data/{folder_name}"

    ### 1.读取配置文件
    yaml_path = f"{expt_path}/workflow_config_master_Alpha158_{market_name}_{d_model}.yaml"
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    # universe = config["market"]


    ### 2.实验参数【修改到config中】
    seed_num = config["task"]["model"]["kwargs"]["seed_num"]
    d_feat = config["task"]["model"]["kwargs"]["d_feat"]                                        # d_feat = 174    # 158
    d_model = config["task"]["model"]["kwargs"]["d_model"]                                      # d_model = 256
    t_nhead = config["task"]["model"]["kwargs"]["t_nhead"]                                      # t_nhead = 4
    s_nhead = config["task"]["model"]["kwargs"]["s_nhead"]                                      # s_nhead = 2
    dropout = config["task"]["model"]["kwargs"]["dropout"]                                      # dropout = 0.2  # 0.5
    gate_input_start_index = config["task"]["model"]["kwargs"]["gate_input_start_index"]        # gate_input_start_index = 174    #158
    gate_input_end_index = config["task"]["model"]["kwargs"]["gate_input_end_index"]            # gate_input_end_index = 237      # 221

    n_epoch = config["task"]["model"]["kwargs"]["n_epochs"]                                      # n_epoch = 100
    lr = config["task"]["model"]["kwargs"]["lr"]                                                # lr = 1e-5
    GPU = config["task"]["model"]["kwargs"]["GPU"]                                              # GPU = 0
    train_stop_loss_thred = config["task"]["model"]["kwargs"]["train_stop_loss_thred"]          # train_stop_loss_thred = 0.92

    beta = config["task"]["model"]["kwargs"]["beta"]                                            # csi300:5;csi500:3;csi800:2;else:2

    benchmark =  config["benchmark"]
    backday = config['task']['dataset']['kwargs']['step_len']



    ### 3.读取实验数据
    # data_dir = f'../Data/Results/{folder_name}'

    with open(f'{expt_path}/{market_name}_self_dl_train.pkl', 'rb') as f:
        dl_train = pickle.load(f)
    with open(f'{expt_path}/{market_name}_self_dl_valid.pkl', 'rb') as f:
        dl_valid = pickle.load(f)
    with open(f'{expt_path}/{market_name}_self_dl_test.pkl', 'rb') as f:
        dl_test = pickle.load(f)
    print("Data Loaded.")

    ### 4. 实验：输出d_model特征值
    # for seed in [0]:
    model = MASTERModel(
            d_feat = d_feat, d_model = d_model, t_nhead = t_nhead, s_nhead = s_nhead, T_dropout_rate=dropout, S_dropout_rate=dropout,
            beta=beta, gate_input_end_index=gate_input_end_index, gate_input_start_index=gate_input_start_index,
            n_epochs=n_epoch, lr = lr, GPU = GPU, seed = seed, train_stop_loss_thred = train_stop_loss_thred,
            save_path='model/', save_prefix=market_name
        )


    param_path = f'{expt_path}/{market_name}_backday_{backday}_self_exp_{seed}_{step}_{d_model}.pkl'
    old_state_dict = torch.load(param_path)

    new_keys = ["encoder.0.weight", "encoder.0.bias", "encoder.1.pe", "encoder.2.qtrans.weight", "encoder.2.ktrans.weight", "encoder.2.vtrans.weight", "encoder.2.norm1.weight", "encoder.2.norm1.bias", "encoder.2.norm2.weight", "encoder.2.norm2.bias", "encoder.2.ffn.0.weight", "encoder.2.ffn.0.bias", "encoder.2.ffn.3.weight", "encoder.2.ffn.3.bias", "encoder.3.qtrans.weight", "encoder.3.ktrans.weight", "encoder.3.vtrans.weight", "encoder.3.norm1.weight", "encoder.3.norm1.bias", "encoder.3.norm2.weight", "encoder.3.norm2.bias", "encoder.3.ffn.0.weight", "encoder.3.ffn.0.bias", "encoder.3.ffn.3.weight", "encoder.3.ffn.3.bias", "encoder.4.trans.weight", "decoder.weight", "decoder.bias"]
    old_keys = ["layers.0.weight", "layers.0.bias", "layers.1.pe", "layers.2.qtrans.weight", "layers.2.ktrans.weight", "layers.2.vtrans.weight", "layers.2.norm1.weight", "layers.2.norm1.bias", "layers.2.norm2.weight", "layers.2.norm2.bias", "layers.2.ffn.0.weight", "layers.2.ffn.0.bias", "layers.2.ffn.3.weight", "layers.2.ffn.3.bias", "layers.3.qtrans.weight", "layers.3.ktrans.weight", "layers.3.vtrans.weight", "layers.3.norm1.weight", "layers.3.norm1.bias", "layers.3.norm2.weight", "layers.3.norm2.bias", "layers.3.ffn.0.weight", "layers.3.ffn.0.bias", "layers.3.ffn.3.weight", "layers.3.ffn.3.bias", "layers.4.trans.weight", "layers.5.weight", "layers.5.bias"]
    key_tsfm = {old_key: new_key for old_key, new_key in zip(old_keys, new_keys)}
    new_state_dict = {key_tsfm.get(k, k): v for k, v in old_state_dict.items()}
    model.model.load_state_dict(new_state_dict)

    ### 5. 生成并保存d_model特征值数据
    # train
    df_train = model.encode(dl_train)

    features1_train = dl_train.data
    features1_train.columns = features1_train.columns.get_level_values(-1)
    features1_train = features1_train.reset_index()

    features2_train = df_train.reset_index()
    features_train = pd.merge(features1_train, features2_train, on=['datetime', 'instrument'], how='right')
    print(f'Train data completed...')

    # valie
    df_valid = model.encode(dl_valid)

    features1_valid = dl_valid.data
    features1_valid.columns = features1_valid.columns.get_level_values(-1)
    features1_valid = features1_valid.reset_index()
    features2_valid = df_valid.reset_index()
    features_valid = pd.merge(features1_valid, features2_valid, on=['datetime', 'instrument'], how='right')
    print(f'Valid data completed...')

    # test
    df_test = model.encode(dl_test)

    features1_test = dl_test.data
    features1_test.columns = features1_test.columns.get_level_values(-1)
    features1_test = features1_test.reset_index()
    features2_test = df_test.reset_index()
    features_test = pd.merge(features1_test, features2_test, on=['datetime', 'instrument'], how='right')
    print(f'Test data completed...')

    # concat
    features_data = pd.concat([features_train, features_valid, features_test], ignore_index=True)

    # 拼接adjclose
    init(provider_uri=qlib_path, region=REG_CN)
    from qlib.data import D
    instruments = features_data['instrument'].drop_duplicates().tolist()
    df_adjclose = D.features(instruments, ["$adjclose"],
                          start_time=features_data['datetime'].min(),
                          end_time=features_data['datetime'].max())
    df_adjclose = df_adjclose.reset_index()
    features_data = pd.merge(features_data, df_adjclose, on=['instrument', 'datetime'], how='left')

    # 保存
    features_data = features_data.drop_duplicates(keep='last')
    features_filename = f"{expt_path}/{market_name}_merge_exp_data_{seed}_{step}_encode_{d_model}.parquet"
    features_data.to_parquet(features_filename)

    print(f'Completed...')

if __name__ == "__main__":
    fire.Fire(main)