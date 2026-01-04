from master_freq import MASTERModel
import os
import pickle
import numpy as np
import time
import fire
import pandas as pd
import yaml
import torch
from pathlib import Path

# 配置文件路径
config_file = "./workflow_config_master_Alpha158.yaml"
print(f"使用配置文件: {config_file}")

with open(config_file, 'r') as f:
    config = yaml.safe_load(f)
    
universe = config["market"]  # 从配置文件取值
backday = config['task']['dataset']['kwargs']['step_len']
prefix = 'opensource'

predict_data_dir = f'data/self_exp/opensource'
with open(f'{predict_data_dir}/{universe}_self_dl_test.pkl', 'rb') as f:
    dl_test = pickle.load(f)
    
print("Test Data Loaded.")

# 模型参数
d_feat = 310  # 根据实际数据特征维度调整
d_model = 256
t_nhead = 4
s_nhead = 2      # 增加空间注意力头
dropout = 0.2    # 降低dropout
gate_input_start_index = 310  # 根据实际数据调整
gate_input_end_index = 373   # 根据实际数据调整

n_epoch = 100
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
freq_enhance_type = "parallel"  # 'joint', 'gate', 'parallel'

# 损失函数配置
use_msic_loss = False        # 启用多任务IC损失
use_adaptive_loss = True    # 启用自适应损失功能
mse_weight = 1.0            # MSE损失权重
ic_weight = 0.5             # IC损失权重
nonlinear_weight = 0.01   # 非线性损失权重
use_nonlinear = True       # 启用非线性损失功能

# 读取因子
save_path = './Backtest_results/predictions'
os.makedirs(save_path, exist_ok=True)

factor_path = './Master_results'
factor_folder = Path(factor_path)
factor_files = [file for file in factor_folder.glob("*self_exp*.pkl")]  # 获取所有匹配的文件

# 新老模型参数映射（根据master2.py的结构更新）
new_keys = [
    "feature_gate.trans.weight", "feature_gate.trans.bias",
    "encoder_layers.0.weight", "encoder_layers.0.bias",
    "encoder_layers.1.pe", 
    "encoder_layers.2.qtrans.weight", "encoder_layers.2.ktrans.weight", "encoder_layers.2.vtrans.weight", 
    "encoder_layers.2.norm1.weight", "encoder_layers.2.norm1.bias", 
    "encoder_layers.2.norm2.weight", "encoder_layers.2.norm2.bias", 
    "encoder_layers.2.ffn.0.weight", "encoder_layers.2.ffn.0.bias", 
    "encoder_layers.2.ffn.3.weight", "encoder_layers.2.ffn.3.bias",
    "encoder_layers.3.qtrans.weight", "encoder_layers.3.ktrans.weight", "encoder_layers.3.vtrans.weight", 
    "encoder_layers.3.norm1.weight", "encoder_layers.3.norm1.bias", 
    "encoder_layers.3.norm2.weight", "encoder_layers.3.norm2.bias", 
    "encoder_layers.3.ffn.0.weight", "encoder_layers.3.ffn.0.bias", 
    "encoder_layers.3.ffn.3.weight", "encoder_layers.3.ffn.3.bias",
    "encoder_layers.4.trans.weight",
    "decoder.weight", "decoder.bias"
]

old_keys = [
    "feature_gate.trans.weight", "feature_gate.trans.bias",
    "layers.0.weight", "layers.0.bias",
    "layers.1.pe", 
    "layers.2.qtrans.weight", "layers.2.ktrans.weight", "layers.2.vtrans.weight", 
    "layers.2.norm1.weight", "layers.2.norm1.bias", 
    "layers.2.norm2.weight", "layers.2.norm2.bias", 
    "layers.2.ffn.0.weight", "layers.2.ffn.0.bias", 
    "layers.2.ffn.3.weight", "layers.2.ffn.3.bias",
    "layers.3.qtrans.weight", "layers.3.ktrans.weight", "layers.3.vtrans.weight", 
    "layers.3.norm1.weight", "layers.3.norm1.bias", 
    "layers.3.norm2.weight", "layers.3.norm2.bias", 
    "layers.3.ffn.0.weight", "layers.3.ffn.0.bias", 
    "layers.3.ffn.3.weight", "layers.3.ffn.3.bias",
    "layers.4.trans.weight",
    "layers.5.weight", "layers.5.bias"
]

key_tsfm = {old_key: new_key for old_key, new_key in zip(old_keys, new_keys)}

# 频域增强相关的键映射（如果存在）
frequency_keys_mapping = {
    "frequency_enhance.gate.0.weight": "frequency_enhance.gate.0.weight",
    "frequency_enhance.gate.0.bias": "frequency_enhance.gate.0.bias",
    "frequency_enhance.freq_transform.weight": "frequency_enhance.freq_transform.weight",
    "frequency_enhance.freq_transform.bias": "frequency_enhance.freq_transform.bias",
    "frequency_enhance.joint_transform.weight": "frequency_enhance.joint_transform.weight",
    "frequency_enhance.joint_transform.bias": "frequency_enhance.joint_transform.bias"
}

key_tsfm.update(frequency_keys_mapping)

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
        
        # 转换键名
        new_state_dict = {}
        for k, v in old_state_dict.items():
            new_key = key_tsfm.get(k, k)
            new_state_dict[new_key] = v
        
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
            save_path=save_path, 
            save_prefix=universe,
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
        
        # 加载权重
        model.model.load_state_dict(new_state_dict)
        model.fitted = 1
        
        # 设置模型为评估模式
        model.model.eval()
        
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
        import traceback
        traceback.print_exc()
        continue

# 保存所有测试指标结果
test_process_info.to_csv(f'{save_path}/test_metrics_results.csv', index=False)
print(f"所有处理完成! 共处理了 {len(test_process_info)} 个模型")
print(f"测试指标已保存到: {save_path}/test_metrics_results.csv")

# 输出统计信息
if not test_process_info.empty:
    print("\n测试结果统计:")
    print(f"平均IC: {test_process_info['Test_IC'].mean():.4f} ± {test_process_info['Test_IC'].std():.4f}")
    print(f"平均RIC: {test_process_info['Test_RIC'].mean():.4f} ± {test_process_info['Test_RIC'].std():.4f}")
    print(f"最佳IC: {test_process_info['Test_IC'].max():.4f} (Seed={test_process_info.loc[test_process_info['Test_IC'].idxmax(), 'Seed']})")
    print(f"最佳RIC: {test_process_info['Test_RIC'].max():.4f} (Seed={test_process_info.loc[test_process_info['Test_RIC'].idxmax(), 'Seed']})")