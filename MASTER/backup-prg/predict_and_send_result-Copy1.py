from master import MASTERModel
import pickle
import numpy as np
import time
import pandas as pd
from datetime import datetime
import os
import yaml
from send_predictions_by_email import send_email_via_126


# Please install qlib first before load the data.
# with open("./workflow_config_master_Alpha158_4_last_day_predictions.yaml", 'r') as f:
with open("./workflow_config_master_Alpha158_4_last_day_predictions_csi800.yaml", 'r') as f:
# with open("./workflow_config_master_Alpha158_4_last_day_predictions_csi1000.yaml", 'r') as f:
    config = yaml.safe_load(f)
    
universe = config["market"] # 优化，直接从配置文件取值
prefix = 'opensource'

predict_data_dir = f'data/daily_predict/opensource'
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

n_epoch = 60
lr = 1e-5
GPU = 0
train_stop_loss_thred = 0.94

if universe == 'csi300':
    beta = 5
elif universe == 'csi500':
    beta = 3
elif universe == 'csi800':
    beta = 2
elif universe == 'csi1000':
    beta = 2
else:
    beta = 2


import sys

# added by xhy
if '--enable_rank_loss' in sys.argv:
    print('Rank loss enabled!')
    enable_rank_loss = True
    universe = universe + '_rank'
else:
    enable_rank_loss = False
    
# Load and Test
######################################################################################
# 选择效果最好的种子模型
if universe == 'csi300':
    seed = 2
elif universe == 'csi500':
    seed = 0
elif universe == 'csi800':
    seed = 4
elif universe == 'csi1000':
    seed = 0
else:
    seed = 0 

param_path = f'model/20250228_train_from_2008_to_2022/{universe}_{prefix}_self_exp_{seed}_{seed}.pkl'

print(f'Model Loaded from {param_path}')
model = MASTERModel(
        d_feat = d_feat, d_model = d_model, t_nhead = t_nhead, s_nhead = s_nhead, T_dropout_rate=dropout, S_dropout_rate=dropout,
        beta=beta, gate_input_end_index=gate_input_end_index, gate_input_start_index=gate_input_start_index,
        n_epochs=n_epoch, lr = lr, GPU = GPU, seed = seed, train_stop_loss_thred = train_stop_loss_thred,
        save_path='model/', save_prefix=universe, enable_rank_loss=enable_rank_loss,
    )

model.load_param(param_path)
predictions, metrics = model.predict(dl_test)
print(predictions)
print(metrics)

# 以dataframe格式保存预测结果
pred_frame = predictions.to_frame()
pred_frame.columns = ['score']
pred_frame.reset_index(inplace=True)
pred_frame.to_csv(f'master_predictions_{universe}_{seed}.csv', index=False, date_format='%Y-%m-%d')

# 单独保存一份轻量级的最近单日评分，用于单日选股操作
last_day = pred_frame['datetime'].max().strftime('%Y-%m-%d')
file_path = f'master_predictions_{universe}_{seed}_single_day_{last_day}.csv'
# 如果文件已存在，说明之前已经生成过该日期的评分文件，则不再重新生成和发送（应对节假日数据源也更新文件，但最新日期仍是上个交易日的情况）
if os.path.isfile(file_path):
    print(f"{file_path} 已存在（已发送过），无需重新生成和发送")
else:
    # 如不存在，则重新生成和发送
    single_day_pred = pred_frame[pred_frame['datetime'] == last_day]
    single_day_pred.to_csv(file_path, index=False, date_format='%Y-%m-%d')
    print(single_day_pred)

    # # 发送单日评分文件
    # send_email_via_126(
    #     sender_email="mfzabc@126.com",   # 你的126邮箱地址
    #     sender_auth_code="RGcbJUE72j5SEtZu",  # 在126邮箱设置的授权码
    #     receiver_email="278857448@qq.com",
    #     csv_file_path=file_path,
    #     subject="MASTER股票评分_" + universe,
    #     body="附件为最新MASTER股票评分，请查收。" # 自定义正文
    # )

    # # 发送单日评分文件
    # send_email_via_126(
    #     sender_email="mfzabc@126.com",   # 你的126邮箱地址
    #     sender_auth_code="RGcbJUE72j5SEtZu",  # 在126邮箱设置的授权码
    #     receiver_email="14721846@qq.com",
    #     csv_file_path=file_path,
    #     subject="MASTER股票评分_" + universe,
    #     body="附件为最新MASTER股票评分，请查收。" # 自定义正文
    # )
