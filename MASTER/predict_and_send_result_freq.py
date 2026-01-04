from master_freq import MASTERModel  # 修改为导入master2
import pickle
import numpy as np
import time
import pandas as pd
from datetime import datetime
import os
import sys
import yaml
from send_predictions_by_email import send_email_via_126


# 从命令行参数获取 universe
if len(sys.argv) > 1:
    universe_arg = sys.argv[1]
else:
    universe_arg = "csi800"  # 默认值

# 根据 universe 参数选择对应的 YAML 文件
if universe_arg == "csi800":
    config_file = "./workflow_config_master_Alpha158_4_last_day_predictions_csi800.yaml"
elif universe_arg == "csi1000":
    config_file = "./workflow_config_master_Alpha158_4_last_day_predictions_csi1000.yaml"
elif universe_arg == "csi300":
    config_file = "./workflow_config_master_Alpha158_4_last_day_predictions_csi300.yaml"
else:
    config_file = "./workflow_config_master_Alpha158_4_last_day_predictions.yaml"

print(f"使用配置文件: {config_file}")

with open(config_file, 'r') as f:
    config = yaml.safe_load(f)
    
universe = config["market"] # 优化，直接从配置文件取值
prefix = 'opensource'

predict_data_dir = f'data/daily_predict/opensource'
with open(f'{predict_data_dir}/{universe}_self_dl_test.pkl', 'rb') as f:
    dl_test = pickle.load(f)
    
print("Data Loaded.")

# 模型参数
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
freq_enhance_type = "joint"  # 'joint', 'gate', 'parallel'

# 损失函数配置
use_msic_loss = True
use_adaptive_loss = False   # 预测时关闭自适应
mse_weight = 1.0
ic_weight = 0.1
nonlinear_weight = 0.005
use_nonlinear = False       # 预测时关闭非线性损失

# 排名损失选项
if '--enable_rank_loss' in sys.argv:
    print('Rank loss enabled!')
    enable_rank_loss = True
    universe_suffix = universe + '_rank'
else:
    enable_rank_loss = False
    universe_suffix = universe
    
# Load and Test
######################################################################################
# 选择效果最好的种子模型
if universe == 'csi300':
    seed = 0
elif universe == 'csi500':
    seed = 0
elif universe == 'csi800':
    seed = 2
elif universe == 'csi1000':
    seed = 0
else:
    seed = 0 

param_path = f'model/20250228_train_from_2008_to_2022/{universe}_{prefix}_self_exp_{seed}_{seed}.pkl'

print(f'Model Loaded from {param_path}')

# 初始化模型（包含频域增强和损失函数配置）
model = MASTERModel(
    d_feat=d_feat, 
    d_model=d_model, 
    t_nhead=t_nhead, 
    s_nhead=s_nhead, 
    T_dropout_rate=dropout, 
    S_dropout_rate=dropout,
    beta=beta, 
    gate_input_end_index=gate_input_end_index, 
    gate_input_start_index=gate_input_start_index,
    n_epochs=n_epoch, 
    lr=lr, 
    GPU=GPU, 
    seed=seed, 
    train_stop_loss_thred=train_stop_loss_thred,
    save_path='model/', 
    save_prefix=universe_suffix,  # 使用带后缀的universe名称
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

# 加载模型参数
model.load_param(param_path)

# 进行预测
predictions, metrics = model.predict(dl_test)
print(predictions)
print(metrics)

# 以dataframe格式保存预测结果
pred_frame = predictions.to_frame()
pred_frame.columns = ['score']
pred_frame.reset_index(inplace=True)

# 确保datetime列存在
if 'datetime' not in pred_frame.columns:
    # 尝试找到时间列
    time_cols = [col for col in pred_frame.columns if col.lower() in ['date', 'time', 'timestamp']]
    if time_cols:
        pred_frame = pred_frame.rename(columns={time_cols[0]: 'datetime'})
    else:
        # 如果索引是时间类型
        if isinstance(pred_frame.index, pd.DatetimeIndex):
            pred_frame = pred_frame.reset_index()
            if 'index' in pred_frame.columns:
                pred_frame = pred_frame.rename(columns={'index': 'datetime'})

pred_frame.to_csv(f'master_predictions_{universe_suffix}_{seed}.csv', index=False, date_format='%Y-%m-%d')

# 单独保存一份轻量级的最近单日评分，用于单日选股操作
last_day = pred_frame['datetime'].max().strftime('%Y-%m-%d')
file_path = f'master_predictions_{universe_suffix}_{seed}_single_day_{last_day}.csv'

# 如果文件已存在，说明之前已经生成过该日期的评分文件，则不再重新生成和发送
if os.path.isfile(file_path):
    print(f"{file_path} 已存在（已发送过），无需重新生成和发送")
else:
    # 如不存在，则重新生成和发送
    single_day_pred = pred_frame[pred_frame['datetime'] == last_day]
    single_day_pred.to_csv(file_path, index=False, date_format='%Y-%m-%d')
    print(single_day_pred)
    
    # 发送邮件（如果需要）
    try:
        # 这里可以添加发送邮件的代码
        # send_email_via_126(file_path, f"Master预测结果 - {universe_suffix} - {last_day}")
        print(f"预测结果已保存到 {file_path}")
    except Exception as e:
        print(f"发送邮件时出错: {e}")

# 输出模型配置信息
print("\n模型配置信息:")
print(f"频域增强: {use_frequency} ({freq_enhance_type})")
print(f"多任务损失: {use_msic_loss}")
print(f"自适应损失: {use_adaptive_loss}")
print(f"非线性损失: {use_nonlinear}")
print(f"损失权重 - MSE: {mse_weight}, IC: {ic_weight}, Nonlinear: {nonlinear_weight}")