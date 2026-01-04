from master import MASTERModel
import pickle
import numpy as np
import time
import pandas as pd
import yaml
import torch

# Please install qlib first before load the data.
with open("./workflow_config_master_Alpha158.yaml", 'r') as f:
    config = yaml.safe_load(f)

universe = 'csi300' # 优化，直接从配置文件取值
backday = 8
prefix = 'extend_5' # ['original','opensource'], which training data are you using
test_data_dir = f'data/self_exp'

with open(f'{test_data_dir}/{prefix}/{universe}_extend_dl_test.pkl', 'rb') as f:
    dl_test = pickle.load(f)

print("Data Loaded.")


# 需要与训练代码一致
d_feat = 158
d_model = 256
t_nhead = 4
s_nhead = 2
dropout = 0.5
gate_input_start_index = 158
gate_input_end_index = 221

if universe == 'csi300':
    beta = 5
elif universe == 'csi500':
    beta = 3
elif universe == 'csi800':
    beta = 2
else:
    beta = 2



# 不需要设置这些值
n_epoch = 1
lr = 1e-5
GPU = 0
train_stop_loss_thred = 0.95

for seed in [0, 1, 2, 3, 4][:1]:
    model = MASTERModel(
            d_feat = d_feat, d_model = d_model, t_nhead = t_nhead, s_nhead = s_nhead, T_dropout_rate=dropout, S_dropout_rate=dropout,
            beta=beta, gate_input_end_index=gate_input_end_index, gate_input_start_index=gate_input_start_index,
            n_epochs=n_epoch, lr = lr, GPU = GPU, seed = seed, train_stop_loss_thred = train_stop_loss_thred,
            save_path='model/', save_prefix=universe
        )
    param_path = f'model/{universe}_{prefix}_backday_{backday}_self_exp_{seed}_{seed}.pkl' # 不知道为啥，实际文件有两个seed
    old_state_dict = torch.load(param_path)

    new_keys = ["encoder.0.weight", "encoder.0.bias", "encoder.1.pe", "encoder.2.qtrans.weight", "encoder.2.ktrans.weight", "encoder.2.vtrans.weight", "encoder.2.norm1.weight", "encoder.2.norm1.bias", "encoder.2.norm2.weight", "encoder.2.norm2.bias", "encoder.2.ffn.0.weight", "encoder.2.ffn.0.bias", "encoder.2.ffn.3.weight", "encoder.2.ffn.3.bias", "encoder.3.qtrans.weight", "encoder.3.ktrans.weight", "encoder.3.vtrans.weight", "encoder.3.norm1.weight", "encoder.3.norm1.bias", "encoder.3.norm2.weight", "encoder.3.norm2.bias", "encoder.3.ffn.0.weight", "encoder.3.ffn.0.bias", "encoder.3.ffn.3.weight", "encoder.3.ffn.3.bias", "encoder.4.trans.weight", "decoder.weight", "decoder.bias"]
    old_keys = ["layers.0.weight", "layers.0.bias", "layers.1.pe", "layers.2.qtrans.weight", "layers.2.ktrans.weight", "layers.2.vtrans.weight", "layers.2.norm1.weight", "layers.2.norm1.bias", "layers.2.norm2.weight", "layers.2.norm2.bias", "layers.2.ffn.0.weight", "layers.2.ffn.0.bias", "layers.2.ffn.3.weight", "layers.2.ffn.3.bias", "layers.3.qtrans.weight", "layers.3.ktrans.weight", "layers.3.vtrans.weight", "layers.3.norm1.weight", "layers.3.norm1.bias", "layers.3.norm2.weight", "layers.3.norm2.bias", "layers.3.ffn.0.weight", "layers.3.ffn.0.bias", "layers.3.ffn.3.weight", "layers.3.ffn.3.bias", "layers.4.trans.weight", "layers.5.weight", "layers.5.bias"]
    key_tsfm = {old_key: new_key for old_key, new_key in zip(old_keys, new_keys)}

    new_state_dict = {key_tsfm.get(k, k): v for k, v in old_state_dict.items()}

    model.model.load_state_dict(new_state_dict)
    model.fitted = 1
    df = model.encode(dl_test)
    df.to_csv(f'out_features/{universe}_{prefix}_backday_{backday}_self_exp_{seed}_{seed}.csv')
    print(f'Complete on seed {seed}.')

    predictions, metrics = model.predict(dl_test)
    pred_frame = predictions.to_frame()
    pred_frame.columns = ['score']
    pred_frame.reset_index(inplace=True)
    pred_frame.to_csv(f'master_predictions_backday_{backday}_{universe}_{seed}.csv', index=False, date_format='%Y-%m-%d')

