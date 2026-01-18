import pickle
import yaml
import fire
import sys
import os
import torch
from pathlib import Path

dirname = "/home/idc2/notebook/quant"
if os.path.exists(dirname):
    sys.path.append(dirname)

from master.master import MASTERModel
##########################################################

### 2.读取因子
def read_param_filename(param_path, market_name):
    param_path = Path(param_path)
    param_filename_list = [file.name for file in param_path.glob(f"{market_name}*self_exp*.pkl")]
    return param_filename_list


def processing_pred(
    # proj_path: str = f"c:/Quant",
    # data_path: str = f"C:/data_set",
    market_name: str = "csi800",
    data_path: str=f"/home/a/notebook/zxf/data/Daily_data/Good_seed/seed2",
    is_batch: bool = False
):

    # notebook/zxf/data/Daily_data/Good_seed/seed2/csi800_backday_8_self_exp_12_54.pkl
    # data_path="/home/a/notebook/zxf/data/Daily_data"
    # folder_name="seed2"
    
    ### 1. 设置目录/文件名
    # config_filename = f'{data_path}/Good_seed/{folder_name}/workflow_config_master_Alpha158_{market_name}.yaml'
    # testdata_filename = f'{data_path}/Training_data/{market_name}/{market_name}_self_dl_test.pkl'

    # param_file_path =  Path(f'{data_path}/Good_seed/{folder_name}')
    # param_filename_list = [file.name for file in param_file_path.glob(f"{market_name}*self_exp*.pkl")]

    config_filename = f'{data_path}/workflow_config_master_Alpha158_{market_name}.yaml'
    testdata_filename = f'{data_path}/{market_name}_self_dl_test.pkl'

    if is_batch:
        param_filename_list = read_param_filename(f'{data_path}/Master_results', market_name)
        save_path = f'{data_path}/Backtest_Results/predictions'
        os.makedirs(save_path, exist_ok=True)

    else:
        param_file_path =  Path(f'{data_path}')
        param_filename_list = [file.name for file in param_file_path.glob(f"{market_name}*self_exp*.pkl")]
        save_path = data_path
        

    ### 2. 读取参数
    with open(config_filename, 'r') as f:
        config = yaml.safe_load(f)

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

    ### 3. 读取测试数据集
    with open(testdata_filename, 'rb') as f:
        dl_test = pickle.load(f)

    # #######################################################################
    # ### 4. 运行实验,并生成预测值


    for filename in param_filename_list:
        seed = int(filename.split('_')[5].split('.pkl')[0])
        step = int(filename.split('_')[6].split('.pkl')[0])
        print(f"filename:{filename} / seed:{seed} / step:{step}")

        model = MASTERModel(
                d_feat = d_feat, d_model = d_model, t_nhead = t_nhead, s_nhead = s_nhead, T_dropout_rate=dropout, S_dropout_rate=dropout,
                beta=beta, gate_input_end_index=gate_input_end_index, gate_input_start_index=gate_input_start_index,
                n_epochs=n_epoch, lr = lr, GPU = GPU, seed = seed, train_stop_loss_thred = train_stop_loss_thred,
                save_path=save_path, save_prefix=market_name
            )
        param_file_path =  Path(f'{data_path}') / filename
        
        old_state_dict = torch.load(param_file_path)

        # new_keys = ["encoder.0.weight", "encoder.0.bias", "encoder.1.pe", "encoder.2.qtrans.weight", "encoder.2.ktrans.weight", "encoder.2.vtrans.weight", "encoder.2.norm1.weight", "encoder.2.norm1.bias", "encoder.2.norm2.weight", "encoder.2.norm2.bias", "encoder.2.ffn.0.weight", "encoder.2.ffn.0.bias", "encoder.2.ffn.3.weight", "encoder.2.ffn.3.bias", "encoder.3.qtrans.weight", "encoder.3.ktrans.weight", "encoder.3.vtrans.weight", "encoder.3.norm1.weight", "encoder.3.norm1.bias", "encoder.3.norm2.weight", "encoder.3.norm2.bias", "encoder.3.ffn.0.weight", "encoder.3.ffn.0.bias", "encoder.3.ffn.3.weight", "encoder.3.ffn.3.bias", "encoder.4.trans.weight", "decoder.weight", "decoder.bias"]
        # old_keys = ["layers.0.weight", "layers.0.bias", "layers.1.pe", "layers.2.qtrans.weight", "layers.2.ktrans.weight", "layers.2.vtrans.weight", "layers.2.norm1.weight", "layers.2.norm1.bias", "layers.2.norm2.weight", "layers.2.norm2.bias", "layers.2.ffn.0.weight", "layers.2.ffn.0.bias", "layers.2.ffn.3.weight", "layers.2.ffn.3.bias", "layers.3.qtrans.weight", "layers.3.ktrans.weight", "layers.3.vtrans.weight", "layers.3.norm1.weight", "layers.3.norm1.bias", "layers.3.norm2.weight", "layers.3.norm2.bias", "layers.3.ffn.0.weight", "layers.3.ffn.0.bias", "layers.3.ffn.3.weight", "layers.3.ffn.3.bias", "layers.4.trans.weight", "layers.5.weight", "layers.5.bias"]
        # key_tsfm = {old_key: new_key for old_key, new_key in zip(old_keys, new_keys)}

        # new_state_dict = {key_tsfm.get(k, k): v for k, v in old_state_dict.items()}

        model.model.load_state_dict(old_state_dict)
        model.fitted = 1

        predictions, metrics = model.predict(dl_test)
        pred_frame = predictions.to_frame()
        pred_frame.columns = ['score']
        pred_frame.reset_index(inplace=True)
        pred_filename = f'{save_path}/master_predictions_backday_{backday}_{market_name}_{seed}_{step}.csv'
        pred_frame.to_csv(pred_filename, index=False, date_format='%Y-%m-%d')

if __name__ == "__main__":
    fire.Fire(processing_pred)
