from master import MASTERModel
import pickle
import numpy as np
import time
import pandas as pd
import yaml

# Please install qlib first before load the data.
with open("./workflow_config_master_Alpha158.yaml", 'r') as f:
    config = yaml.safe_load(f)


universe = config["market"] # 优化，直接从配置文件取值
prefix = 'opensource' # ['original','opensource'], which training data are you using
train_data_dir = f'data/self_exp'
# print(f'{train_data_dir}/{prefix}/{universe}_dl_train.pkl')
# with open(f'{train_data_dir}/{prefix}/{universe}_extend_dl_train.pkl', 'rb') as f:
with open(f'{train_data_dir}/{prefix}/{universe}_self_dl_train.pkl', 'rb') as f:
    dl_train = pickle.load(f)

predict_data_dir = f'data/self_exp/{prefix}'
# with open(f'{predict_data_dir}/{universe}_extend_dl_valid.pkl', 'rb') as f:
with open(f'{predict_data_dir}/{universe}_self_dl_valid.pkl', 'rb') as f:
    dl_valid = pickle.load(f)
# with open(f'{predict_data_dir}/{universe}_extend_dl_test.pkl', 'rb') as f:
with open(f'{predict_data_dir}/{universe}_self_dl_test.pkl', 'rb') as f:
    dl_test = pickle.load(f)

    
# test = pd.read_pickle(f'{predict_data_dir}/{universe}_dl_test.pkl')
# print(test.data)

print("Data Loaded.")


d_feat = 158
d_model = 256
t_nhead = 4
s_nhead = 2
dropout = 0.5
gate_input_start_index = 158
gate_input_end_index = 221

# # luo add
# d_feat = 158
# d_model = 384
# t_nhead = 6
# s_nhead = 4
# dropout = 0.3
# gate_input_start_index = 158
# gate_input_end_index = 221

if universe == 'csi300':
    beta = 5
elif universe == 'csi500':
    beta = 3
elif universe == 'csi800':
    beta = 2
else:
    beta = 2

n_epoch = 60
lr = 1e-5
GPU = 0
train_stop_loss_thred = 0.945


ic = []
icir = []
ric = []
ricir = []

backday = config['task']['dataset']['kwargs']['step_len']

# Training
######################################################################################
# for seed in [0, 1, 2, 3, 4]:
for seed in [1]:
    model = MASTERModel(
        d_feat = d_feat, d_model = d_model, t_nhead = t_nhead, s_nhead = s_nhead, T_dropout_rate=dropout, S_dropout_rate=dropout,
        beta=beta, gate_input_end_index=gate_input_end_index, gate_input_start_index=gate_input_start_index,
        n_epochs=n_epoch, lr = lr, GPU = GPU, seed = seed, train_stop_loss_thred = train_stop_loss_thred,
        save_path='model', save_prefix=f'{universe}_{prefix}_backday_{backday}_self_exp_{seed}'
    )

    start = time.time()
    # Train
    model.fit(dl_train, dl_valid)

    print("Model Trained.")

    # Test
    predictions, metrics = model.predict(dl_test)

    # predictions.to_csv('master_predictions_csi300.csv')
    pred_frame = predictions.to_frame()
    pred_frame.columns = ['score']
    pred_frame.reset_index(inplace=True)
    pred_frame.to_csv(f'master_predictions_backday_{backday}_{universe}_{seed}.csv', index=False, date_format='%Y-%m-%d')
    
    running_time = time.time()-start
    
    print('Seed: {:d} time cost : {:.2f} sec'.format(seed, running_time))
    print(metrics)
    print(predictions)

    
    

    ic.append(metrics['IC'])
    icir.append(metrics['ICIR'])
    ric.append(metrics['RIC'])
    ricir.append(metrics['RICIR'])
######################################################################################

# Load and Test
######################################################################################
# for seed in [0, 1, 2, 3, 4]:
#     param_path = f'model/{universe}_{prefix}_self_exp_{seed}_{seed}.pkl'
#     # param_path = f'/Users/Carmelo/Documents/GitHub/qlib/examples/benchmarks/MASTER/model/csi300master_0.pkl'

#     print(f'Model Loaded from {param_path}')
#     model = MASTERModel(
#             d_feat = d_feat, d_model = d_model, t_nhead = t_nhead, s_nhead = s_nhead, T_dropout_rate=dropout, S_dropout_rate=dropout,
#             beta=beta, gate_input_end_index=gate_input_end_index, gate_input_start_index=gate_input_start_index,
#             n_epochs=n_epoch, lr = lr, GPU = GPU, seed = seed, train_stop_loss_thred = train_stop_loss_thred,
#             save_path='model/', save_prefix=universe
#         )
#     model.load_param(param_path)
#     predictions, metrics = model.predict(dl_test)
#     print(metrics)

#     ic.append(metrics['IC'])
#     icir.append(metrics['ICIR'])
#     ric.append(metrics['RIC'])
#     ricir.append(metrics['RICIR'])
    
######################################################################################

print("IC: {:.4f} pm {:.4f}".format(np.mean(ic), np.std(ic)))
print("ICIR: {:.4f} pm {:.4f}".format(np.mean(icir), np.std(icir)))
print("RIC: {:.4f} pm {:.4f}".format(np.mean(ric), np.std(ric)))
print("RICIR: {:.4f} pm {:.4f}".format(np.mean(ricir), np.std(ricir)))