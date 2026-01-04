from master_freq import MASTERModel
import pickle
import numpy as np
import time
import os
import sys
import pandas as pd
import yaml
from pathlib import Path

# 配置文件路径
config_path = "./workflow_config_master_Alpha158.yaml"

# 确保使用绝对路径
current_dir = Path(__file__).parent
with open(current_dir / config_path, 'r') as f:
    config = yaml.safe_load(f)
    
universe = config["market"]
data_dir = current_dir / 'data/self_exp/opensource'

# 确保数据目录存在
data_dir.mkdir(parents=True, exist_ok=True)

# 加载数据
try:
    with open(data_dir / f'{universe}_self_dl_train.pkl', 'rb') as f:
        dl_train = pickle.load(f)
    with open(data_dir / f'{universe}_self_dl_valid.pkl', 'rb') as f:
        dl_valid = pickle.load(f)
    with open(data_dir / f'{universe}_self_dl_test.pkl', 'rb') as f:
        dl_test = pickle.load(f)
    print("Data Loaded Successfully.")
except FileNotFoundError as e:
    print(f"Error loading data: {e}")
    print("Please check the data directory and file names.")
    sys.exit(1)



# 模型参数
d_feat = 217  # 根据实际数据特征维度调整
d_model = 256
t_nhead = 4
s_nhead = 2      # 增加空间注意力头
dropout = 0.3    # 降低dropout
gate_input_start_index = 217  # 根据实际数据调整
gate_input_end_index = 373   # 根据实际数据调整

self.d_gate_input = (gate_input_end_index - gate_input_start_index)
self.d_src = self.gate_input_start_index

n_epoch = 200
lr = 1e-5       # 保持优化后的学习率
GPU = 0
train_stop_loss_thred = 0.80

# Beta参数设置
beta_config = {
    'csi300': 5,
    'csi500': 3,
    'csi800': 2,
    'csi1000': 2
}
beta = beta_config.get(universe, 2)

# 频域增强配置
use_frequency = True  # 启用频域增强
freq_enhance_type = "gate"  # 'joint', 'gate', 'parallel'

# 损失函数配置
use_msic_loss = True        # 启用多任务IC损失
use_adaptive_loss = True    # 启用自适应损失功能
mse_weight = 1.0            # MSE损失权重
ic_weight = 0.5             # IC损失权重
nonlinear_weight = 0.01   # 非线性损失权重
use_nonlinear = True       # 启用非线性损失功能

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
save_path = current_dir / 'Master_results'
save_path.mkdir(exist_ok=True)

# 训练过程记录
train_process_info = pd.DataFrame([])

# 训练循环 - 使用多个种子提高稳定性
# seeds = [0, 1, 2, 3, 4]  # 使用多个种子
seeds = [1]   # 使用多个种子

for seed in seeds:
    print(f"Training with seed {seed}...")
    
    model = MASTERModel(
        d_feat=d_feat, d_model=d_model, t_nhead=t_nhead, s_nhead=s_nhead, 
        T_dropout_rate=dropout, S_dropout_rate=dropout,
        beta=beta, gate_input_end_index=gate_input_end_index, 
        gate_input_start_index=gate_input_start_index,
        n_epochs=n_epoch, lr=lr, GPU=GPU, seed=seed, 
        train_stop_loss_thred=train_stop_loss_thred,
        save_path=str(save_path), 
        save_prefix=f'{universe_suffix}_backday_{backday}_self_exp_{seed}',
        enable_rank_loss=enable_rank_loss,
        # 新增频域增强参数
        use_frequency=use_frequency,
        freq_enhance_type=freq_enhance_type,
        # 新增损失函数参数
        use_msic_loss=use_msic_loss,
        use_adaptive_loss=use_adaptive_loss,
        mse_weight=mse_weight,
        ic_weight=ic_weight,
        nonlinear_weight=nonlinear_weight,
        use_nonlinear=use_nonlinear
    )

    start = time.time()
    
    # 训练模型
    try:
        train_process_info_df = model.fit(dl_train, dl_valid)
        train_process_info_df['Seed'] = seed
        train_process_info_df = train_process_info_df[['Seed', 'Step', 'Train_loss', 
                                                      'Valid_IC', 'Valid_ICIR', 'Valid_RIC', 'Valid_RICIR']]
        train_process_info = pd.concat([train_process_info, train_process_info_df], ignore_index=True)

        print(f"Seed {seed} Model Trained Successfully.")

        # 测试模型
        predictions, metrics = model.predict(dl_test)

        # 保存预测结果
        pred_frame = predictions.to_frame()
        pred_frame.columns = ['score']
        pred_frame.reset_index(inplace=True)
        pred_frame.to_csv(
            save_path / f'master_predictions_backday_{backday}_{universe_suffix}_{seed}.csv', 
            index=False, date_format='%Y-%m-%d'
        )

        running_time = time.time() - start
        print(f'Seed {seed} time cost: {running_time:.2f} sec')
        print(f"Metrics: {metrics}")

        # 记录结果
        seed_list.append(seed)
        ic.append(metrics['IC'])
        icir.append(metrics['ICIR'])
        ric.append(metrics['RIC'])
        ricir.append(metrics['RICIR'])
        
    except Exception as e:
        print(f"Error training model with seed {seed}: {e}")
        continue

# 保存训练和测试结果
if not train_process_info.empty:
    train_process_info.to_csv(save_path / 'train_metrics_results.csv', index=False)

if seed_list:  # 如果有成功运行的种子
    ic_data = {
        'SEED': seed_list,
        'IC': ic,
        'ICIR': icir,
        'RIC': ric,
        'RICIR': ricir
    }
    df = pd.DataFrame(ic_data)
    df.to_csv(save_path / 'test_metrics_results.csv', index=False)

    # 输出最终结果
    print("\nFinal Results:")
    print(f"IC: {np.mean(ic):.4f} ± {np.std(ic):.4f}")
    print(f"ICIR: {np.mean(icir):.4f} ± {np.std(icir):.4f}")
    print(f"RIC: {np.mean(ric):.4f} ± {np.std(ric):.4f}")
    print(f"RICIR: {np.mean(ricir):.4f} ± {np.std(ricir):.4f}")
else:
    print("No models were successfully trained.")