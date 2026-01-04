from master import MASTERModel
import os
import pickle
import numpy as np
import time
import fire
import pandas as pd
import yaml
import torch
from pathlib import Path



# 根据 universe 参数选择对应的 YAML 文件
# if universe_arg == "csi800":
#     config_file = "./workflow_config_master_Alpha158_4_last_day_predictions_csi800.yaml"
# elif universe_arg == "csi1000":
#     config_file = "./workflow_config_master_Alpha158_4_last_day_predictions_csi1000.yaml"
# else:
#     config_file = "./workflow_config_master_Alpha158_4_last_day_predictions.yaml"

config_file = "./workflow_config_master_Alpha158.yaml"
print(f"使用配置文件: {config_file}")

with open(config_file, 'r') as f:
    config = yaml.safe_load(f)
    
universe = config["market"] # 优化，直接从配置文件取值
backday = config['task']['dataset']['kwargs']['step_len']
prefix = 'opensource'

predict_data_dir = f'data/self_exp/opensource'
with open(f'{predict_data_dir}/{universe}_self_dl_test.pkl', 'rb') as f:
    dl_test = pickle.load(f)
    
# test = pd.read_pickle(f'{predict_data_dir}/{universe}_dl_test.pkl')
# print(test.data)

print("Test Data Loaded.")

# 模型参数
d_feat = 158  # 根据实际数据特征维度调整
d_model = 256
t_nhead = 4
s_nhead = 2      # 增加空间注意力头
dropout = 0.3    # 降低dropout
gate_input_start_index = 158  # 根据实际数据调整
gate_input_end_index = 221   # 根据实际数据调整

n_epoch = 100
lr = 1e-5
GPU = 0
train_stop_loss_thred = 0.95  # 可根据需要调整

# Beta参数设置
beta_config = {
    'csi300': 5,
    'csi500': 3,
    'csi800': 2,
    'csi1000': 2
}
beta = beta_config.get(universe, 2)




# 读取因子
save_path = './Backtest_results/predictions'
os.makedirs(save_path, exist_ok=True)

factor_path = './Master_results'
factor_folder = Path(factor_path)
factor_files = [file for file in factor_folder.glob("*self_exp*.pkl")]  # 获取所有匹配的文件

# 新老模型参数映射
new_keys = ["encoder.0.weight", "encoder.0.bias", "encoder.1.pe", "encoder.2.qtrans.weight", "encoder.2.ktrans.weight", "encoder.2.vtrans.weight", "encoder.2.norm1.weight", "encoder.2.norm1.bias", "encoder.2.norm2.weight", "encoder.2.norm2.bias", "encoder.2.ffn.0.weight", "encoder.2.ffn.0.bias", "encoder.2.ffn.3.weight", "encoder.2.ffn.3.bias", "encoder.3.qtrans.weight", "encoder.3.ktrans.weight", "encoder.3.vtrans.weight", "encoder.3.norm1.weight", "encoder.3.norm1.bias", "encoder.3.norm2.weight", "encoder.3.norm2.bias", "encoder.3.ffn.0.weight", "encoder.3.ffn.0.bias", "encoder.3.ffn.3.weight", "encoder.3.ffn.3.bias", "encoder.4.trans.weight", "decoder.weight", "decoder.bias"]
old_keys = ["layers.0.weight", "layers.0.bias", "layers.1.pe", "layers.2.qtrans.weight", "layers.2.ktrans.weight", "layers.2.vtrans.weight", "layers.2.norm1.weight", "layers.2.norm1.bias", "layers.2.norm2.weight", "layers.2.norm2.bias", "layers.2.ffn.0.weight", "layers.2.ffn.0.bias", "layers.2.ffn.3.weight", "layers.2.ffn.3.bias", "layers.3.qtrans.weight", "layers.3.ktrans.weight", "layers.3.vtrans.weight", "layers.3.norm1.weight", "layers.3.norm1.bias", "layers.3.norm2.weight", "layers.3.norm2.bias", "layers.3.ffn.0.weight", "layers.3.ffn.0.bias", "layers.3.ffn.3.weight", "layers.3.ffn.3.bias", "layers.4.trans.weight", "layers.5.weight", "layers.5.bias"]
key_tsfm = {old_key: new_key for old_key, new_key in zip(old_keys, new_keys)}

# 初始化结果DataFrame
test_process_info = pd.DataFrame(columns=['Seed', 'Step', 'Test_IC', 'Test_ICIR', 'Test_RIC', 'Test_RICIR'])

# 循环处理每个PKL文件
for factor_file in factor_files:
    filename = factor_file.name
    
    # 解析文件名获取seed和step
    parts = filename.split('.')[0].split('_')
    if len(parts) >= 7:  # 确保有足够的部分
        seed, step = parts[5], parts[6]  # 第6和第7个部分（索引5和6）
        seed, step = int(seed), int(step)
        print(f"处理文件: {filename} | Seed: {seed} | Step: {step}")
    else:
        print(f"无法解析文件名: {filename}, 跳过此文件")
        continue
    
    try:
        # 加载模型状态字典
        old_state_dict = torch.load(factor_file)
        new_state_dict = {key_tsfm.get(k, k): v for k, v in old_state_dict.items()}
        
        # 初始化模型并加载权重
        model = MASTERModel(
            d_feat=d_feat, d_model=d_model, t_nhead=t_nhead, s_nhead=s_nhead, 
            T_dropout_rate=dropout, S_dropout_rate=dropout,
            beta=beta, gate_input_end_index=gate_input_end_index, 
            gate_input_start_index=gate_input_start_index,
            n_epochs=n_epoch, lr=lr, GPU=GPU, seed=seed, 
            train_stop_loss_thred=train_stop_loss_thred,
            save_path=save_path, save_prefix=universe
        )
        model.model.load_state_dict(new_state_dict)
        model.fitted = 1
        
        # 进行预测
        predictions, metrics = model.predict(dl_test)
        
        # 记录测试指标
        df = {
            'Seed': seed,
            'Step': step,
            'Test_IC': metrics['IC'],
            'Test_ICIR': metrics['ICIR'],
            'Test_RIC': metrics['RIC'],
            'Test_RICIR': metrics['RICIR']
        }
        test_process_info = pd.concat([test_process_info, pd.DataFrame([df])], ignore_index=True)
        
        # 保存预测结果到CSV
        pred_frame = predictions.to_frame()
        pred_frame.columns = ['score']
        pred_frame.reset_index(inplace=True)
        pred_frame.to_csv(
            f'{save_path}/master_predictions_backday_{backday}_{universe}_{seed}_{step}.csv', 
            index=False, 
            date_format='%Y-%m-%d'
        )
        
        print(f"完成处理: Seed={seed}, Step={step}, IC={metrics['IC']:.4f}")
        
    except Exception as e:
        print(f"处理文件 {filename} 时出错: {str(e)}")
        continue

# 保存所有测试指标结果
test_process_info.to_csv(f'{save_path}/test_metrics_results.csv', index=False)
print(f"所有处理完成! 共处理了 {len(test_process_info)} 个模型")
print(f"测试指标已保存到: {save_path}/test_metrics_results.csv")