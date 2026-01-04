from master import MASTERModel
import pickle
import numpy as np
import time
import os
import sys
import pandas as pd
import yaml

# 配置文件路径
config_path = "./workflow_config_master_Alpha158.yaml"  # 可根据需要修改

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
    
universe = config["market"]
data_dir = f'data/self_exp/opensource'  # 可根据需要修改为绝对路径

# 加载数据
with open(f'{data_dir}/{universe}_self_dl_train.pkl', 'rb') as f:
    dl_train = pickle.load(f)
with open(f'{data_dir}/{universe}_self_dl_valid.pkl', 'rb') as f:
    dl_valid = pickle.load(f)
with open(f'{data_dir}/{universe}_self_dl_test.pkl', 'rb') as f:
    dl_test = pickle.load(f)

print("Data Loaded.")

# 模型参数
d_feat = 435  # 根据实际数据特征维度调整
d_model = 256
t_nhead = 4
s_nhead = 2      # 增加空间注意力头
dropout = 0.3    # 降低dropout
gate_input_start_index = 435  # 根据实际数据调整
gate_input_end_index = 523   # 根据实际数据调整

n_epoch = 100
lr = 1e-5
GPU = 0
train_stop_loss_thred = 0.92  # 可根据需要调整

# Beta参数设置
beta_config = {
    'csi300': 5,
    'csi500': 3,
    'csi800': 2,
    'csi1000': 2
}
beta = beta_config.get(universe, 2)

# 排名损失选项
if '--enable_rank_loss' in sys.argv:
    print('Rank loss enabled!')
    enable_rank_loss = True
    universe_suffix = universe + '_rank'
else:
    enable_rank_loss = False
    universe_suffix = universe

# 初始化结果列表
ic, icir, ric, ricir, seed_list = [], [], [], [], []
backday = config['task']['dataset']['kwargs']['step_len']

# 创建保存目录
save_path = f'Master_results'
os.makedirs(save_path, exist_ok=True)

# 训练过程记录
train_process_info = pd.DataFrame([])

# 训练循环
# for seed in [0, 1, 2, 3, 4]:  # 可根据需要调整种子
for seed in [0]:   
    model = MASTERModel(
        d_feat=d_feat, d_model=d_model, t_nhead=t_nhead, s_nhead=s_nhead, 
        T_dropout_rate=dropout, S_dropout_rate=dropout,
        beta=beta, gate_input_end_index=gate_input_end_index, 
        gate_input_start_index=gate_input_start_index,
        n_epochs=n_epoch, lr=lr, GPU=GPU, seed=seed, 
        train_stop_loss_thred=train_stop_loss_thred,
        save_path=save_path, 
        save_prefix=f'{universe_suffix}_backday_{backday}_self_exp_{seed}',
        enable_rank_loss=enable_rank_loss
    )

    start = time.time()
    
    # 训练模型
    train_process_info_df = model.fit(dl_train, dl_valid)
    train_process_info_df['Seed'] = seed
    train_process_info_df = train_process_info_df[['Seed', 'Step', 'Train_loss', 
                                                  'Valid_IC', 'Valid_ICIR', 'Valid_RIC', 'Valid_RICIR']]
    train_process_info = pd.concat([train_process_info, train_process_info_df], ignore_index=True)

    print(f"{seed} Model Trained.")

    # 测试模型
    predictions, metrics = model.predict(dl_test)

    # 保存预测结果
    pred_frame = predictions.to_frame()
    pred_frame.columns = ['score']
    pred_frame.reset_index(inplace=True)
    pred_frame.to_csv(f'{save_path}/master_predictions_backday_{backday}_{universe_suffix}_{seed}.csv', 
                     index=False, date_format='%Y-%m-%d')

    running_time = time.time() - start
    print(f'Seed: {seed} time cost: {running_time:.2f} sec')
    print(metrics)

    # 记录结果
    seed_list.append(seed)
    ic.append(metrics['IC'])
    icir.append(metrics['ICIR'])
    ric.append(metrics['RIC'])
    ricir.append(metrics['RICIR'])

# 保存训练和测试结果
train_process_info.to_csv(f'{save_path}/train_metrics_results.csv', index=False)

ic_data = {
    'SEED': seed_list,
    'IC': ic,
    'ICIR': icir,
    'RIC': ric,
    'RICIR': ricir
}
df = pd.DataFrame(ic_data)
df.to_csv(f'{save_path}/test_metrics_results.csv', index=False)

# 输出最终结果
print(f"IC: {np.mean(ic):.4f} pm {np.std(ic):.4f}")
print(f"ICIR: {np.mean(icir):.4f} pm {np.std(icir):.4f}")
print(f"RIC: {np.mean(ric):.4f} pm {np.std(ric):.4f}")
print(f"RICIR: {np.mean(ricir):.4f} pm {np.std(ricir):.4f}")