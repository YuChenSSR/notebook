import torch
from torch import nn
from torch.nn.modules.linear import Linear
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.normalization import LayerNorm
import math
import numpy as np

from base_model import SequenceModel

# 频域增强模块
class FrequencyEnhancement(nn.Module):
    def __init__(self, d_model, enhance_type="gate"):
        super(FrequencyEnhancement, self).__init__()
        self.enhance_type = enhance_type
        self.d_model = d_model
        
        if enhance_type == "gate":
            self.gate = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.Sigmoid()
            )
        elif enhance_type == "parallel":
            self.freq_transform = nn.Linear(d_model, d_model)
        elif enhance_type == "joint":
            self.joint_transform = nn.Linear(d_model * 2, d_model)
    
    def time_to_frequency(self, x):
        # 将时序信号转换到频域
        # x shape: [batch_size, seq_len, d_model]
        x_fft = torch.fft.rfft(x, dim=1)
        magnitude = torch.abs(x_fft)
        phase = torch.angle(x_fft)
        return magnitude, phase
    
    def frequency_to_time(self, magnitude, phase):
        # 将频域信号转换回时域
        real = magnitude * torch.cos(phase)
        imag = magnitude * torch.sin(phase)
        x_complex = torch.complex(real, imag)
        x_reconstructed = torch.fft.irfft(x_complex, n=x_complex.shape[1], dim=1)
        return x_reconstructed
    
    def forward(self, x):
        if self.enhance_type == "none" or not self.enhance_type:
            return x
        
        # 检查输入维度
        if len(x.shape) == 2:
            # 如果是2D输入，增加序列维度
            x = x.unsqueeze(1)
        
        batch_size, seq_len, d_model = x.shape
        
        # 转换到频域
        magnitude, phase = self.time_to_frequency(x)
        
        if self.enhance_type == "gate":
            # 使用门控机制融合时域和频域信息
            magnitude_mean = torch.mean(magnitude, dim=1, keepdim=True)
            phase_mean = torch.mean(phase, dim=1, keepdim=True)
            
            freq_features = torch.cat([magnitude_mean, phase_mean], dim=-1)
            gate_weights = self.gate(freq_features)
            enhanced = x * gate_weights
            
        elif self.enhance_type == "parallel":
            # 并行处理时域和频域信息
            freq_processed = self.freq_transform(magnitude)
            enhanced = x + self.frequency_to_time(freq_processed, phase)
            
        elif self.enhance_type == "joint":
            # 联合处理时域和频域信息
            joint_features = torch.cat([x, magnitude], dim=-1)
            enhanced = self.joint_transform(joint_features)
        else:
            enhanced = x
        
        return enhanced.squeeze(1) if enhanced.shape[1] == 1 else enhanced

# 自适应损失权重管理器
class AdaptiveLossWeights:
    def __init__(self, initial_weights, adaptation_rate=0.1, min_weight=0.01, max_weight=2.0):
        self.weights = initial_weights.copy()
        self.initial_weights = initial_weights.copy()
        self.adaptation_rate = adaptation_rate
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.loss_history = {key: [] for key in initial_weights.keys()}
        self.performance_history = []
    
    def update_weights_based_performance(self, current_performance):
        """基于性能指标调整权重"""
        # 简单的性能驱动调整策略
        if len(self.performance_history) > 1:
            last_performance = self.performance_history[-1]
            if current_performance > last_performance:
                # 性能提升，稍微降低IC权重以避免过拟合
                self.weights['ic'] = max(self.min_weight, self.weights['ic'] * 0.95)
            else:
                # 性能下降，恢复IC权重
                self.weights['ic'] = min(self.max_weight, self.weights['ic'] * 1.05)
        
        self.performance_history.append(current_performance)
        return self.weights
    
    def update_weights_based_loss(self, current_losses):
        """基于损失值调整权重"""
        for key in current_losses:
            self.loss_history[key].append(current_losses[key])
        
        if len(self.loss_history['mse']) > 10:  # 有足够历史数据后开始调整
            # 计算最近损失的相对变化
            recent_mse = np.mean(self.loss_history['mse'][-5:])
            recent_ic = np.mean(self.loss_history['ic'][-5:]) if self.loss_history['ic'] else 0
            
            if recent_ic > 0 and recent_mse > 0:
                # 根据损失比例调整权重
                loss_ratio = recent_ic / recent_mse
                if loss_ratio > 2.0:  # IC损失相对过大
                    self.weights['ic'] = max(self.min_weight, self.weights['ic'] * 0.9)
                elif loss_ratio < 0.5:  # IC损失相对过小
                    self.weights['ic'] = min(self.max_weight, self.weights['ic'] * 1.1)
        
        return self.weights
    
    def reset(self):
        """重置到初始权重"""
        self.weights = self.initial_weights.copy()
        self.loss_history = {key: [] for key in self.initial_weights.keys()}
        self.performance_history = []

# 多任务损失函数
class MultiTaskLoss(nn.Module):
    def __init__(self, mse_weight=1.0, ic_weight=0.1, nonlinear_weight=0.005, 
                 use_msic=True, use_nonlinear=False, use_adaptive=False):
        super(MultiTaskLoss, self).__init__()
        self.mse_weight = mse_weight
        self.ic_weight = ic_weight
        self.nonlinear_weight = nonlinear_weight
        self.use_msic = use_msic
        self.use_nonlinear = use_nonlinear
        self.use_adaptive = use_adaptive
        self.mse_loss = nn.MSELoss()
        
        # 初始化自适应权重管理器
        if use_adaptive:
            initial_weights = {
                'mse': mse_weight,
                'ic': ic_weight,
                'nonlinear': nonlinear_weight
            }
            self.adaptive_manager = AdaptiveLossWeights(initial_weights)
    
    def information_coefficient_loss(self, predictions, targets):
        """计算IC损失"""
        # 确保输入是1D张量
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        pred_mean = predictions - torch.mean(predictions)
        target_mean = targets - torch.mean(targets)
        covariance = torch.mean(pred_mean * target_mean)
        pred_std = torch.std(predictions)
        target_std = torch.std(targets)
        
        if pred_std == 0 or target_std == 0:
            return torch.tensor(0.0, device=predictions.device)
        
        ic = covariance / (pred_std * target_std)
        return 1.0 - torch.abs(ic)  # 最大化绝对IC值
    
    def nonlinear_correlation_loss(self, predictions, targets):
        """计算非线性相关性损失（HSIC）"""
        try:
            # 确保正确的维度
            predictions = predictions.view(-1, 1)
            targets = targets.view(-1, 1)
            
            pred_centered = predictions - torch.mean(predictions)
            target_centered = targets - torch.mean(targets)
            
            # 使用RBF核
            sigma = 1.0
            pred_sq_dist = torch.cdist(pred_centered, pred_centered, p=2)
            target_sq_dist = torch.cdist(target_centered, target_centered, p=2)
            
            pred_kernel = torch.exp(-pred_sq_dist / (2 * sigma ** 2))
            target_kernel = torch.exp(-target_sq_dist / (2 * sigma ** 2))
            
            n = predictions.size(0)
            if n <= 1:
                return torch.tensor(0.0, device=predictions.device)
                
            H = torch.eye(n, device=predictions.device) - torch.ones(n, n, device=predictions.device) / n
            hsic = torch.trace(torch.mm(torch.mm(pred_kernel, H), torch.mm(target_kernel, H))) / ((n - 1) ** 2)
            
            return -hsic  # 最大化HSIC
            
        except Exception as e:
            # 如果计算失败，返回0损失
            return torch.tensor(0.0, device=predictions.device)
    
    def update_adaptive_weights(self, current_losses=None, current_performance=None):
        """更新自适应权重"""
        if not self.use_adaptive:
            return
        
        if current_performance is not None:
            self.adaptive_manager.update_weights_based_performance(current_performance)
        elif current_losses is not None:
            self.adaptive_manager.update_weights_based_loss(current_losses)
        
        # 更新当前权重
        self.mse_weight = self.adaptive_manager.weights['mse']
        self.ic_weight = self.adaptive_manager.weights['ic']
        self.nonlinear_weight = self.adaptive_manager.weights['nonlinear']
    
    def get_current_weights(self):
        """获取当前权重"""
        return {
            'mse': self.mse_weight,
            'ic': self.ic_weight,
            'nonlinear': self.nonlinear_weight
        }
    
    def forward(self, predictions, targets, current_performance=None):
        # 计算各项损失
        mse_loss = self.mse_loss(predictions, targets)
        total_loss = self.mse_weight * mse_loss
        
        current_losses = {'mse': mse_loss.item()}
        
        if self.use_msic:
            ic_loss = self.information_coefficient_loss(predictions, targets)
            total_loss += self.ic_weight * ic_loss
            current_losses['ic'] = ic_loss.item()
        
        if self.use_nonlinear:
            nonlinear_loss = self.nonlinear_correlation_loss(predictions, targets)
            total_loss += self.nonlinear_weight * nonlinear_loss
            current_losses['nonlinear'] = nonlinear_loss.item()
        
        # 更新自适应权重
        if self.use_adaptive:
            self.update_adaptive_weights(current_losses, current_performance)
        
        return total_loss

# PositionalEncoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:x.shape[1], :]

# SAttention
class SAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.temperature = math.sqrt(self.d_model/nhead)

        self.qtrans = nn.Linear(d_model, d_model, bias=False)
        self.ktrans = nn.Linear(d_model, d_model, bias=False)
        self.vtrans = nn.Linear(d_model, d_model, bias=False)

        attn_dropout_layer = []
        for i in range(nhead):
            attn_dropout_layer.append(Dropout(p=dropout))
        self.attn_dropout = nn.ModuleList(attn_dropout_layer)

        # input LayerNorm
        self.norm1 = LayerNorm(d_model, eps=1e-5)

        # FFN layerNorm
        self.norm2 = LayerNorm(d_model, eps=1e-5)
        self.ffn = nn.Sequential(
            Linear(d_model, d_model),
            nn.ReLU(),
            Dropout(p=dropout),
            Linear(d_model, d_model),
            Dropout(p=dropout)
        )

    def forward(self, x):
        x = self.norm1(x)
        q = self.qtrans(x).transpose(0,1)
        k = self.ktrans(x).transpose(0,1)
        v = self.vtrans(x).transpose(0,1)

        dim = int(self.d_model/self.nhead)
        att_output = []
        for i in range(self.nhead):
            if i==self.nhead-1:
                qh = q[:, :, i * dim:]
                kh = k[:, :, i * dim:]
                vh = v[:, :, i * dim:]
            else:
                qh = q[:, :, i * dim:(i + 1) * dim]
                kh = k[:, :, i * dim:(i + 1) * dim]
                vh = v[:, :, i * dim:(i + 1) * dim]

            atten_ave_matrixh = torch.softmax(torch.matmul(qh, kh.transpose(1, 2)) / self.temperature, dim=-1)
            if self.attn_dropout:
                atten_ave_matrixh = self.attn_dropout[i](atten_ave_matrixh)
            att_output.append(torch.matmul(atten_ave_matrixh, vh).transpose(0, 1))
        att_output = torch.concat(att_output, dim=-1)

        # FFN
        xt = x + att_output
        xt = self.norm2(xt)
        att_output = xt + self.ffn(xt)

        return att_output

# TAttention
class TAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.qtrans = nn.Linear(d_model, d_model, bias=False)
        self.ktrans = nn.Linear(d_model, d_model, bias=False)
        self.vtrans = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout = []
        if dropout > 0:
            for i in range(nhead):
                self.attn_dropout.append(Dropout(p=dropout))
            self.attn_dropout = nn.ModuleList(self.attn_dropout)

        # input LayerNorm
        self.norm1 = LayerNorm(d_model, eps=1e-5)
        # FFN layerNorm
        self.norm2 = LayerNorm(d_model, eps=1e-5)
        # FFN
        self.ffn = nn.Sequential(
            Linear(d_model, d_model),
            nn.ReLU(),
            Dropout(p=dropout),
            Linear(d_model, d_model),
            Dropout(p=dropout)
        )

    def forward(self, x):
        x = self.norm1(x)
        q = self.qtrans(x)
        k = self.ktrans(x)
        v = self.vtrans(x)

        dim = int(self.d_model / self.nhead)
        att_output = []
        for i in range(self.nhead):
            if i==self.nhead-1:
                qh = q[:, :, i * dim:]
                kh = k[:, :, i * dim:]
                vh = v[:, :, i * dim:]
            else:
                qh = q[:, :, i * dim:(i + 1) * dim]
                kh = k[:, :, i * dim:(i + 1) * dim]
                vh = v[:, :, i * dim:(i + 1) * dim]
            atten_ave_matrixh = torch.softmax(torch.matmul(qh, kh.transpose(1, 2)), dim=-1)
            if self.attn_dropout:
                atten_ave_matrixh = self.attn_dropout[i](atten_ave_matrixh)
            att_output.append(torch.matmul(atten_ave_matrixh, vh))
        att_output = torch.concat(att_output, dim=-1)

        # FFN
        xt = x + att_output
        xt = self.norm2(xt)
        att_output = xt + self.ffn(xt)

        return att_output

# Gate
class Gate(nn.Module):
    def __init__(self, d_input, d_output,  beta=1.0):
        super().__init__()
        self.trans = nn.Linear(d_input, d_output)
        self.d_output =d_output
        self.t = beta

    def forward(self, gate_input):
        output = self.trans(gate_input)
        output = torch.softmax(output/self.t, dim=-1)
        return self.d_output*output

# TemporalAttention
class TemporalAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.trans = nn.Linear(d_model, d_model, bias=False)

    def forward(self, z):
        h = self.trans(z) # [N, T, D]
        query = h[:, -1, :].unsqueeze(-1)
        lam = torch.matmul(h, query).squeeze(-1)  # [N, T, D] --> [N, T]
        lam = torch.softmax(lam, dim=1).unsqueeze(1)
        output = torch.matmul(lam, z).squeeze(1)  # [N, 1, T], [N, T, D] --> [N, 1, D]
        return output

# MASTER模型
class MASTER(nn.Module):
    def __init__(self, d_feat, d_model, t_nhead, s_nhead, T_dropout_rate, 
                 S_dropout_rate, gate_input_start_index, gate_input_end_index, 
                 beta, use_frequency=True, freq_enhance_type="gate"):
        super(MASTER, self).__init__()
        # market
        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index
        self.d_gate_input = (gate_input_end_index - gate_input_start_index)
        self.feature_gate = Gate(self.d_gate_input, d_feat, beta=beta)
        
        # 频域增强
        self.use_frequency = use_frequency
        if use_frequency:
            self.frequency_enhance = FrequencyEnhancement(d_model, freq_enhance_type)

        self.encoder_layers = nn.ModuleList([
            nn.Linear(d_feat, d_model),
            PositionalEncoding(d_model),
            TAttention(d_model=d_model, nhead=t_nhead, dropout=T_dropout_rate),
            SAttention(d_model=d_model, nhead=s_nhead, dropout=S_dropout_rate),
            TemporalAttention(d_model=d_model),
        ])
        self.decoder = nn.Linear(d_model, 1)

    def encode(self, x):
        src = x[:, :, :self.gate_input_start_index] # N, T, D
        gate_input = x[:, -1, self.gate_input_start_index:self.gate_input_end_index]
        src = src * torch.unsqueeze(self.feature_gate(gate_input), dim=1)
        
        # 逐层处理
        enc_output = src
        for layer in self.encoder_layers:
            enc_output = layer(enc_output)
        
        # 应用频域增强
        if self.use_frequency:
            enc_output = self.frequency_enhance(enc_output)
            
        return enc_output

    def forward(self, x):
        enc = self.encode(x)
        output = self.decoder(enc)
        output = output.squeeze(-1)
        return output

# MASTERModel
class MASTERModel(SequenceModel):
    def __init__(
            self, d_feat, d_model, t_nhead, s_nhead, gate_input_start_index, 
            gate_input_end_index, T_dropout_rate, S_dropout_rate, beta, 
            use_frequency=True, freq_enhance_type="gate",
            use_msic_loss=True, use_adaptive_loss=False, mse_weight=1.0, 
            ic_weight=0.1, nonlinear_weight=0.005, use_nonlinear=False,
            **kwargs
    ):
        super(MASTERModel, self).__init__(**kwargs)
        self.d_model = d_model
        self.d_feat = d_feat

        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index

        self.T_dropout_rate = T_dropout_rate
        self.S_dropout_rate = S_dropout_rate
        self.t_nhead = t_nhead
        self.s_nhead = s_nhead
        self.beta = beta
        
        # 频域增强参数
        self.use_frequency = use_frequency
        self.freq_enhance_type = freq_enhance_type
        
        # 损失函数参数
        self.use_msic_loss = use_msic_loss
        self.use_adaptive_loss = use_adaptive_loss
        self.mse_weight = mse_weight
        self.ic_weight = ic_weight
        self.nonlinear_weight = nonlinear_weight
        self.use_nonlinear = use_nonlinear

        self.init_model()

    def init_model(self):
        self.model = MASTER(
            d_feat=self.d_feat, 
            d_model=self.d_model, 
            t_nhead=self.t_nhead, 
            s_nhead=self.s_nhead,
            T_dropout_rate=self.T_dropout_rate, 
            S_dropout_rate=self.S_dropout_rate,
            gate_input_start_index=self.gate_input_start_index,
            gate_input_end_index=self.gate_input_end_index, 
            beta=self.beta,
            use_frequency=self.use_frequency,
            freq_enhance_type=self.freq_enhance_type
        )
        
        # 初始化多任务损失函数
        self.criterion = MultiTaskLoss(
            mse_weight=self.mse_weight,
            ic_weight=self.ic_weight,
            nonlinear_weight=self.nonlinear_weight,
            use_msic=self.use_msic_loss,
            use_nonlinear=self.use_nonlinear,
            use_adaptive=self.use_adaptive_loss
        )
        
        super(MASTERModel, self).init_model()
    
    def get_loss(self, pred, target, index=None):
        """重写损失计算函数"""
        # 如果有验证性能信息，可以传递给自适应损失
        return self.criterion(pred, target)
    
    def update_loss_weights(self, current_performance=None):
        """更新损失权重（用于自适应损失）"""
        if hasattr(self.criterion, 'update_adaptive_weights'):
            self.criterion.update_adaptive_weights(current_performance=current_performance)
    
    def get_loss_weights(self):
        """获取当前损失权重"""
        if hasattr(self.criterion, 'get_current_weights'):
            return self.criterion.get_current_weights()
        return {
            'mse': self.mse_weight,
            'ic': self.ic_weight,
            'nonlinear': self.nonlinear_weight
        }