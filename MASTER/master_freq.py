import torch
from torch import nn
from torch.nn.modules.linear import Linear
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.normalization import LayerNorm
import math
import numpy as np

from base_model import SequenceModel


# =========================
# 频域增强模块（P0 改动：irfft 使用原始 seq_len；并保持接口不变）
# =========================
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
        # x: [B, T, D]
        B, T, D = x.shape
        x_fft = torch.fft.rfft(x, dim=1)  # [B, F, D], F = T//2 + 1
        magnitude = torch.abs(x_fft)
        phase = torch.angle(x_fft)
        return magnitude, phase, T

    def frequency_to_time(self, magnitude, phase, T):
        # 使用原始的序列长度 T 进行重构（P0 改动）
        real = magnitude * torch.cos(phase)
        imag = magnitude * torch.sin(phase)
        x_complex = torch.complex(real, imag)  # [B, F, D]
        x_reconstructed = torch.fft.irfft(x_complex, n=T, dim=1)
        return x_reconstructed

    def forward(self, x):
        if self.enhance_type in ("none", None):
            return x

        # 统一为 3D
        if x.dim() == 2:
            x = x.unsqueeze(1)

        B, T, D = x.shape
        magnitude, phase, T0 = self.time_to_frequency(x)

        if self.enhance_type == "gate":
            # 用频域统计做门控
            mag_mean = magnitude.mean(dim=1, keepdim=True)  # [B,1,D]
            ph_mean = phase.mean(dim=1, keepdim=True)       # [B,1,D]
            freq_features = torch.cat([mag_mean, ph_mean], dim=-1)  # [B,1,2D]
            gate_weights = self.gate(freq_features)  # [B,1,D] in [0,1]
            enhanced = x * gate_weights

        elif self.enhance_type == "parallel":
            # 并行：频域线性 -> 回到时域 -> 与原时域相加
            freq_processed = self.freq_transform(magnitude)  # [B,F,D]
            enhanced = x + self.frequency_to_time(freq_processed, phase, T0)

        elif self.enhance_type == "joint":
            # 联合：把频域回到时域后与原时域拼接
            x_freq = self.frequency_to_time(magnitude, phase, T0)  # [B,T,D]
            joint_features = torch.cat([x, x_freq], dim=-1)        # [B,T,2D]
            enhanced = self.joint_transform(joint_features)        # [B,T,D]
        else:
            enhanced = x

        # 维度还原
        return enhanced if enhanced.shape[1] > 1 else enhanced.squeeze(1)


# =========================
# 自适应损失权重管理器（原样）
# =========================
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
        if len(self.performance_history) > 1:
            last_performance = self.performance_history[-1]
            if current_performance > last_performance:
                self.weights['ic'] = max(self.min_weight, self.weights['ic'] * 0.95)
            else:
                self.weights['ic'] = min(self.max_weight, self.weights['ic'] * 1.05)
        self.performance_history.append(current_performance)
        return self.weights

    def update_weights_based_loss(self, current_losses):
        for key in current_losses:
            self.loss_history[key].append(current_losses[key])

        if len(self.loss_history['mse']) > 10:
            recent_mse = np.mean(self.loss_history['mse'][-5:])
            recent_ic = np.mean(self.loss_history['ic'][-5:]) if self.loss_history['ic'] else 0
            if recent_ic > 0 and recent_mse > 0:
                loss_ratio = recent_ic / recent_mse
                if loss_ratio > 2.0:
                    self.weights['ic'] = max(self.min_weight, self.weights['ic'] * 0.9)
                elif loss_ratio < 0.5:
                    self.weights['ic'] = min(self.max_weight, self.weights['ic'] * 1.1)
        return self.weights

    def reset(self):
        self.weights = self.initial_weights.copy()
        self.loss_history = {key: [] for key in self.initial_weights.keys()}
        self.performance_history = []


# =========================
# 多任务损失函数（P0 改动：IC 损失数值安全）
# =========================
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

        if use_adaptive:
            initial_weights = {
                'mse': mse_weight,
                'ic': ic_weight,
                'nonlinear': nonlinear_weight
            }
            self.adaptive_manager = AdaptiveLossWeights(initial_weights)

    def information_coefficient_loss(self, predictions, targets, eps=1e-8):
        # P0 改动：数值安全版
        y = predictions.view(-1)
        t = targets.view(-1)
        y = y - y.mean()
        t = t - t.mean()
        denom = (y.std(unbiased=False) * t.std(unbiased=False)).clamp_min(eps)
        ic = (y * t).mean() / denom
        return 1.0 - ic.abs()

    def nonlinear_correlation_loss(self, predictions, targets):
        # 保持原样（如需降复杂度可在后续 P1 优化）
        try:
            predictions = predictions.view(-1, 1)
            targets = targets.view(-1, 1)

            pred_centered = predictions - torch.mean(predictions)
            target_centered = targets - torch.mean(targets)

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

            return -hsic
        except Exception:
            return torch.tensor(0.0, device=predictions.device)

    def update_adaptive_weights(self, current_losses=None, current_performance=None):
        if not self.use_adaptive:
            return
        if current_performance is not None:
            self.adaptive_manager.update_weights_based_performance(current_performance)
        elif current_losses is not None:
            self.adaptive_manager.update_weights_based_loss(current_losses)
        self.mse_weight = self.adaptive_manager.weights['mse']
        self.ic_weight = self.adaptive_manager.weights['ic']
        self.nonlinear_weight = self.adaptive_manager.weights['nonlinear']

    def get_current_weights(self):
        return {
            'mse': self.mse_weight,
            'ic': self.ic_weight,
            'nonlinear': self.nonlinear_weight
        }

    def forward(self, predictions, targets, current_performance=None):
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

        if self.use_adaptive:
            self.update_adaptive_weights(current_losses, current_performance)

        return total_loss


# =========================
# PositionalEncoding（原样）
# =========================
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


# =========================
# SAttention（原样）
# =========================
class SAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.temperature = math.sqrt(self.d_model / nhead)

        self.qtrans = nn.Linear(d_model, d_model, bias=False)
        self.ktrans = nn.Linear(d_model, d_model, bias=False)
        self.vtrans = nn.Linear(d_model, d_model, bias=False)

        attn_dropout_layer = []
        for _ in range(nhead):
            attn_dropout_layer.append(Dropout(p=dropout))
        self.attn_dropout = nn.ModuleList(attn_dropout_layer)

        self.norm1 = LayerNorm(d_model, eps=1e-5)
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
        q = self.qtrans(x).transpose(0, 1)
        k = self.ktrans(x).transpose(0, 1)
        v = self.vtrans(x).transpose(0, 1)

        dim = int(self.d_model / self.nhead)
        att_output = []
        for i in range(self.nhead):
            if i == self.nhead - 1:
                qh = q[:, :, i * dim:]
                kh = k[:, :, i * dim:]
                vh = v[:, :, i * dim:]
            else:
                qh = q[:, :, i * dim:(i + 1) * dim]
                kh = k[:, :, i * dim:(i + 1) * dim]
                vh = v[:, :, i * dim:(i + 1) * dim]

            # 这里原实现已有 temperature 缩放
            atten_ave_matrixh = torch.softmax(torch.matmul(qh, kh.transpose(1, 2)) / self.temperature, dim=-1)
            if self.attn_dropout:
                atten_ave_matrixh = self.attn_dropout[i](atten_ave_matrixh)
            att_output.append(torch.matmul(atten_ave_matrixh, vh).transpose(0, 1))
        att_output = torch.concat(att_output, dim=-1)

        xt = x + att_output
        xt = self.norm2(xt)
        att_output = xt + self.ffn(xt)

        return att_output


# =========================
# TAttention（P0 改动：加入 1/sqrt(dk) 缩放）
# =========================
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
            for _ in range(nhead):
                self.attn_dropout.append(Dropout(p=dropout))
            self.attn_dropout = nn.ModuleList(self.attn_dropout)

        self.norm1 = LayerNorm(d_model, eps=1e-5)
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
        q = self.qtrans(x)
        k = self.ktrans(x)
        v = self.vtrans(x)

        dim = int(self.d_model / self.nhead)
        dk = float(dim)
        att_output = []
        for i in range(self.nhead):
            if i == self.nhead - 1:
                qh = q[:, :, i * dim:]
                kh = k[:, :, i * dim:]
                vh = v[:, :, i * dim:]
            else:
                qh = q[:, :, i * dim:(i + 1) * dim]
                kh = k[:, :, i * dim:(i + 1) * dim]
                vh = v[:, :, i * dim:(i + 1) * dim]

            # P0 改动：添加缩放
            scores = torch.matmul(qh, kh.transpose(1, 2)) / math.sqrt(dk)
            atten_ave_matrixh = torch.softmax(scores, dim=-1)
            if self.attn_dropout:
                atten_ave_matrixh = self.attn_dropout[i](atten_ave_matrixh)
            att_output.append(torch.matmul(atten_ave_matrixh, vh))
        att_output = torch.concat(att_output, dim=-1)

        xt = x + att_output
        xt = self.norm2(xt)
        att_output = xt + self.ffn(xt)

        return att_output


# =========================
# Gate（P0 改动：不再乘 d_output，且要求输出维度与被门控特征一致）
# =========================
class Gate(nn.Module):
    def __init__(self, d_input, d_output, beta=1.0):
        super().__init__()
        self.trans = nn.Linear(d_input, d_output)
        self.t = beta

    def forward(self, gate_input):
        # gate_input: [N, d_input]
        logits = self.trans(gate_input)  # [N, d_output]
        weights = torch.softmax(logits / self.t, dim=-1)  # [N, d_output], sum=1
        return weights  # 不再乘 d_output


# =========================
# TemporalAttention（原样）
# =========================
class TemporalAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.trans = nn.Linear(d_model, d_model, bias=False)

    def forward(self, z):
        h = self.trans(z)  # [N, T, D]
        query = h[:, -1, :].unsqueeze(-1)
        lam = torch.matmul(h, query).squeeze(-1)  # [N,T]
        lam = torch.softmax(lam, dim=1).unsqueeze(1)
        output = torch.matmul(lam, z).squeeze(1)  # [N, D]
        return output


# =========================
# MASTER（P0 改动：频域增强前置；Gate 维度对齐）
# =========================
class MASTER(nn.Module):
    def __init__(self, d_feat, d_model, t_nhead, s_nhead, T_dropout_rate,
                 S_dropout_rate, gate_input_start_index, gate_input_end_index,
                 beta, use_frequency=True, freq_enhance_type="gate"):
        super(MASTER, self).__init__()

        # gate 切片 & 维度
        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index
        self.d_gate_input = (gate_input_end_index - gate_input_start_index)

        # 被门控的主体特征维度（与 src 最后维度一致）
        self.d_src = self.gate_input_start_index  # src = x[:, :, :d_src]
        assert self.d_src > 0, "gate_input_start_index 必须大于 0"
        assert self.d_gate_input > 0, "gate 区间长度必须大于 0"

        # Gate 输出维度对齐到 src 的特征维度（P0 改动）
        self.feature_gate = Gate(self.d_gate_input, self.d_src, beta=beta)

        # 频域增强
        self.use_frequency = use_frequency
        if use_frequency:
            self.frequency_enhance = FrequencyEnhancement(d_model, freq_enhance_type)

        # 编码层
        self.encoder_layers = nn.ModuleList([
            nn.Linear(self.d_src, d_model),                           # 仅对被门控部分投影
            PositionalEncoding(d_model),
            TAttention(d_model=d_model, nhead=t_nhead, dropout=T_dropout_rate),
            SAttention(d_model=d_model, nhead=s_nhead, dropout=S_dropout_rate),
            TemporalAttention(d_model=d_model),
        ])
        self.decoder = nn.Linear(d_model, 1)

    def encode(self, x):
        # x: [N, T, D_all]
        # 1) 取主干 + 门控
        src = x[:, :, :self.d_src]  # [N,T,d_src]
        gate_input = x[:, -1, self.gate_input_start_index:self.gate_input_end_index]  # [N,d_gate_input]
        gate_w = self.feature_gate(gate_input)  # [N,d_src]
        src = src * gate_w.unsqueeze(1)  # [N,T,d_src]

        # 2) 线性 & 位置编码
        lin = self.encoder_layers[0](src)   # [N,T,D]
        pe = self.encoder_layers[1](lin)    # [N,T,D]

        # 3) 频域增强（P0 改动：前置到注意力之前）
        if self.use_frequency:
            pe = self.frequency_enhance(pe)  # [N,T,D]

        # 4) 注意力堆叠
        out = self.encoder_layers[2](pe)    # TAttention
        out = self.encoder_layers[3](out)   # SAttention
        out = self.encoder_layers[4](out)   # TemporalAttention -> [N,D]

        return out

    def forward(self, x):
        enc = self.encode(x)
        output = self.decoder(enc)  # [N,1]
        output = output.squeeze(-1)
        return output


# =========================
# MASTERModel（保持接口，底层用 P0 修改后的模块）
# =========================
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

        self.use_frequency = use_frequency
        self.freq_enhance_type = freq_enhance_type

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
        return self.criterion(pred, target)

    def update_loss_weights(self, current_performance=None):
        if hasattr(self.criterion, 'update_adaptive_weights'):
            self.criterion.update_adaptive_weights(current_performance=current_performance)

    def get_loss_weights(self):
        if hasattr(self.criterion, 'get_current_weights'):
            return self.criterion.get_current_weights()
        return {
            'mse': self.mse_weight,
            'ic': self.ic_weight,
            'nonlinear': self.nonlinear_weight
        }
