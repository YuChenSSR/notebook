import pandas as pd
from loguru import logger
import pickle
import yaml
import torch
import fire
import sys

from Master.master import MASTERModel

def main(
        market_name: str = "csi800",
        folder_name: str = "csi800_data_merge_20251107",
        data_path: str = "/home/idc2/notebook/zxf/data/master_results",
        seed: int = 4,
        step: int = 22,
):
    ### 1.读取配置文件
    # master_results
    config_filename = f"{data_path}/{folder_name}/workflow_config_master_Alpha158_{market_name}.yaml"
    with open(config_filename, 'r') as f:
        config = yaml.safe_load(f)
    logger.info(f"Config file completed: {config_filename}")
    
    ### 2.读取实验数据
    experiment_dir = f'{data_path}/{folder_name}'

    with open(f'{experiment_dir}/{market_name}_self_dl_train.pkl', 'rb') as f:
        dl_train = pickle.load(f)
    with open(f'{experiment_dir}/{market_name}_self_dl_valid.pkl', 'rb') as f:
        dl_valid = pickle.load(f)
    with open(f'{experiment_dir}/{market_name}_self_dl_test.pkl', 'rb') as f:
        dl_test = pickle.load(f)
    logger.info(f"Data loaded: {experiment_dir}")

    ### 3.实验参数【修改到config中】
    seed_num = config["task"]["model"]["kwargs"]["seed_num"]
    d_feat = config["task"]["model"]["kwargs"]["d_feat"] 
    d_model = config["task"]["model"]["kwargs"]["d_model"] 
    t_nhead = config["task"]["model"]["kwargs"]["t_nhead"] 
    s_nhead = config["task"]["model"]["kwargs"]["s_nhead"]   
    dropout = config["task"]["model"]["kwargs"]["dropout"]  
    gate_input_start_index = config["task"]["model"]["kwargs"]["gate_input_start_index"] 
    gate_input_end_index = config["task"]["model"]["kwargs"]["gate_input_end_index"]  

    n_epoch = config["task"]["model"]["kwargs"]["n_epochs"]  
    lr = config["task"]["model"]["kwargs"]["lr"]      
    GPU = config["task"]["model"]["kwargs"]["GPU"]                                     
    train_stop_loss_thred = config["task"]["model"]["kwargs"]["train_stop_loss_thred"]   

    beta = config["task"]["model"]["kwargs"]["beta"]                         

    benchmark =  config["benchmark"]
    backday = config['task']['dataset']['kwargs']['step_len']


    # added by xhy
    if '--enable_rank_loss' in sys.argv:
        print('Rank loss enabled!')
        enable_rank_loss = True
        universe = universe + '_rank'
    else:
        enable_rank_loss = False

    ### 4. 实验：输出256特征值
    # for seed in [0]:
    model = MASTERModel(
            d_feat = d_feat, d_model = d_model, t_nhead = t_nhead, s_nhead = s_nhead, T_dropout_rate=dropout, S_dropout_rate=dropout,
            beta=beta, gate_input_end_index=gate_input_end_index, gate_input_start_index=gate_input_start_index,
            n_epochs=n_epoch, lr = lr, GPU = GPU, seed = seed, train_stop_loss_thred = train_stop_loss_thred,
            save_path=experiment_dir, save_prefix=market_name
        )
    param_path = f'{experiment_dir}/{market_name}_backday_{backday}_self_exp_{seed}_{step}.pkl'
    old_state_dict = torch.load(param_path)

    new_keys = ["encoder.0.weight", "encoder.0.bias", "encoder.1.pe", "encoder.2.qtrans.weight", "encoder.2.ktrans.weight", "encoder.2.vtrans.weight", "encoder.2.norm1.weight", "encoder.2.norm1.bias", "encoder.2.norm2.weight", "encoder.2.norm2.bias", "encoder.2.ffn.0.weight", "encoder.2.ffn.0.bias", "encoder.2.ffn.3.weight", "encoder.2.ffn.3.bias", "encoder.3.qtrans.weight", "encoder.3.ktrans.weight", "encoder.3.vtrans.weight", "encoder.3.norm1.weight", "encoder.3.norm1.bias", "encoder.3.norm2.weight", "encoder.3.norm2.bias", "encoder.3.ffn.0.weight", "encoder.3.ffn.0.bias", "encoder.3.ffn.3.weight", "encoder.3.ffn.3.bias", "encoder.4.trans.weight", "decoder.weight", "decoder.bias"]
    old_keys = ["layers.0.weight", "layers.0.bias", "layers.1.pe", "layers.2.qtrans.weight", "layers.2.ktrans.weight", "layers.2.vtrans.weight", "layers.2.norm1.weight", "layers.2.norm1.bias", "layers.2.norm2.weight", "layers.2.norm2.bias", "layers.2.ffn.0.weight", "layers.2.ffn.0.bias", "layers.2.ffn.3.weight", "layers.2.ffn.3.bias", "layers.3.qtrans.weight", "layers.3.ktrans.weight", "layers.3.vtrans.weight", "layers.3.norm1.weight", "layers.3.norm1.bias", "layers.3.norm2.weight", "layers.3.norm2.bias", "layers.3.ffn.0.weight", "layers.3.ffn.0.bias", "layers.3.ffn.3.weight", "layers.3.ffn.3.bias", "layers.4.trans.weight", "layers.5.weight", "layers.5.bias"]
    key_tsfm = {old_key: new_key for old_key, new_key in zip(old_keys, new_keys)}
    new_state_dict = {key_tsfm.get(k, k): v for k, v in old_state_dict.items()}
    model.model.load_state_dict(new_state_dict)

    ### 5. 生成并保存256特征值数据
    # train
    df_train = model.encode(dl_train)

    features1_train = dl_train.data
    features1_train.columns = features1_train.columns.get_level_values(-1)
    features1_train = features1_train.reset_index()

    features2_train = df_train.reset_index()
    features_train = pd.merge(features1_train, features2_train, on=['datetime', 'instrument'], how='right')
    logger.info(f"Train data completed...")


    # valid
    df_valid = model.encode(dl_valid)

    features1_valid = dl_valid.data
    features1_valid.columns = features1_valid.columns.get_level_values(-1)
    features1_valid = features1_valid.reset_index()
    features2_valid = df_valid.reset_index()
    features_valid = pd.merge(features1_valid, features2_valid, on=['datetime', 'instrument'], how='right')
    logger.info(f"Valid data completed...")

    # test
    df_test = model.encode(dl_test)

    features1_test = dl_test.data
    features1_test.columns = features1_test.columns.get_level_values(-1)
    features1_test = features1_test.reset_index()
    features2_test = df_test.reset_index()
    features_test = pd.merge(features1_test, features2_test, on=['datetime', 'instrument'], how='right')
    logger.info(f"Test data completed...")

    # concat
    features_data = pd.concat([features_train, features_valid, features_test], ignore_index=True)
    features_filename =f"{data_path}/{folder_name}/{market_name}_merge_exp_data_{seed}_{step}_encode.parquet"
    features_data.to_parquet(features_filename)

    logger.success(f"Data Merge Completed...")

if __name__ == "__main__":
    fire.Fire(main)