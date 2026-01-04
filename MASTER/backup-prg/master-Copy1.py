import torch
import torch.nn as nn
import math
from torch.nn import LayerNorm
from base_model import SequenceModel

class EnhancedMultiHeadAttention(nn.Module):
    """增强版多头注意力：支持多尺度注意力与残差门控"""
    def __init__(self, d_model, nhead, dropout=0.1, scales=[1, 3, 5]):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.scales = scales
        self.nhead = nhead
        self.d_k = d_model // nhead
        
        # 多尺度投影层
        self.qkv_projs = nn.ModuleList([
            nn.Linear(d_model, d_model * 3) for _ in scales
        ])
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm(d_model)
        
        # 门控机制
        self.gate = nn.Sequential(
            nn.Linear(d_model * len(scales), len(scales)),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        B, T, _ = x.shape
        outputs = []
        
        # 多尺度注意力计算
        for i, scale in enumerate(self.scales):
            # 尺度化处理（通过池化调整时间粒度）
            if scale > 1:
                # 使用平均池化，注意：padding确保长度不变
                # 计算需要的padding：当使用奇数尺度的池化时，左右各补 scale//2
                # 这里我们假设使用奇数尺度的scales
                x_pool = nn.AvgPool1d(scale, stride=1, padding=scale//2)(x.transpose(1, 2)).transpose(1, 2)
            else:
                x_pool = x
            
            # 投影QKV
            qkv = self.qkv_projs[i](x_pool).reshape(B, -1, 3, self.nhead, self.d_k)
            q, k, v = qkv.unbind(2)  # [B, T, H, D_k]
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)  # [B, H, T, D_k]
            
            # 缩放点积注意力
            attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
            attn = attn.softmax(dim=-1)
            attn = self.dropout(attn)
            attn_out = (attn @ v).transpose(1, 2).reshape(B, -1, self.d_k * self.nhead)
            attn_out = self.out_proj(attn_out)
            outputs.append(attn_out)
        
        # 多尺度门控融合
        gate_input = torch.cat(outputs, dim=-1)  # [B, T, d_model * num_scales]
        gate_weights = self.gate(gate_input)  # [B, T, num_scales]
        # 将不同尺度的输出按权重加和
        combined = torch.zeros_like(outputs[0])
        for i in range(len(self.scales)):
            combined += gate_weights[..., i].unsqueeze(-1) * outputs[i]
        
        return self.norm(x + combined)

class TemporalHierarchyEncoder(nn.Module):
    """时间层级编码器：捕捉多粒度时序特征"""
    def __init__(self, d_model, nhead, dropout=0.1, scales=[1, 3, 5, 10]):
        super().__init__()
        self.attentions = nn.ModuleList([
            EnhancedMultiHeadAttention(d_model, nhead, dropout, [s]) for s in scales
        ])
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        self.norm = LayerNorm(d_model)

    def forward(self, x):
        # 并行多尺度注意力
        attn_outputs = [attn(x) for attn in self.attentions]
        # 跨尺度特征聚合（取平均）
        h = torch.stack(attn_outputs, dim=-1).mean(dim=-1)
        # 前馈网络
        h = self.ffn(h)
        return self.norm(h + x)  # 残差连接

class MASTER(nn.Module):
    def __init__(
        self,
        d_feat=6,
        d_model=64,
        t_nhead=4,
        s_nhead=4,
        T_dropout=0.1,
        S_dropout=0.1,
        gate_input_start_index=4,
        gate_input_end_index=6,
        beta=1.0,
        time_scales=[1, 3, 5, 10]  # 多尺度时间粒度
    ):
        super().__init__()
        # 保存所有参数为实例属性
        self.d_feat = d_feat
        self.d_model = d_model
        self.t_nhead = t_nhead
        self.s_nhead = s_nhead
        self.T_dropout = T_dropout
        self.S_dropout = S_dropout
        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index
        self.beta = beta
        self.time_scales = time_scales
        
        # 门控机制：输入是门控特征的维度，输出是特征维度d_feat
        self.gate = nn.Sequential(
            nn.Linear(gate_input_end_index - gate_input_start_index, d_feat),
            # 使用温度参数调整softmax的平滑度
            nn.Linear(d_feat, d_feat)  # 先做一次线性变换，然后除以温度
        )
        # 注意：我们需要在forward中手动应用softmax和温度
        
        # 时间层级编码器
        self.time_encoder = TemporalHierarchyEncoder(d_model, t_nhead, T_dropout, time_scales)
        
        # 空间注意力（跨股票）
        self.cross_stock_attn = EnhancedMultiHeadAttention(d_model, s_nhead, S_dropout, scales=[1])
        
        # 时序注意力（用于聚合时间步）
        self.temporal_attn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
            nn.Softmax(dim=1)
        )
        
        # 输出层
        self.output = nn.Linear(d_model, 1)

    def forward(self, x):
        # 获取门控输入：使用当前时间步的门控特征
        gate_in = x[:, -1, self.gate_input_start_index:self.gate_input_end_index]  # [B, gate_dim]
        # 应用门控网络：线性变换 + 温度处理
        gate_logits = self.gate(gate_in)  # [B, d_feat]
        gate_weights = torch.softmax(gate_logits / self.beta, dim=-1)  # [B, d_feat]
        
        # 特征加权（只对特征部分加权）
        src = x[:, :, :self.gate_input_start_index]  # [B, T, d_feat]
        src = src * gate_weights.unsqueeze(1)  # [B, T, d_feat] * [B, 1, d_feat] -> [B, T, d_feat]
        
        # 时间层级编码
        time_feat = self.time_encoder(src)
        
        # 跨股票交互（假设输入已经是同一个batch内不同股票的数据？）
        cross_feat = self.cross_stock_attn(time_feat)
        
        # 时序聚合（每个股票的时间步聚合）
        attn_weights = self.temporal_attn(cross_feat)  # [B, T, 1]
        context = (attn_weights * cross_feat).sum(dim=1)  # [B, d_model]
        
        return self.output(context).squeeze(-1)  # [B]

class MASTERModel(SequenceModel):
    def __init__(self, d_feat, d_model, t_nhead, s_nhead, gate_input_start_index, gate_input_end_index,
            T_dropout_rate, S_dropout_rate, beta, **kwargs):
        super(MASTERModel, self).__init__(**kwargs)
        self.d_model = d_model  # 模型的维度
        self.d_feat = d_feat  # 特征的维度

        self.gate_input_start_index = gate_input_start_index  # 门控输入的起始索引
        self.gate_input_end_index = gate_input_end_index  # 门控输入的结束索引

        self.T_dropout_rate = T_dropout_rate  # 时间注意力机制的Dropout率
        self.S_dropout_rate = S_dropout_rate  # 空间注意力机制的Dropout率
        self.t_nhead = t_nhead  # 时间注意力头的数量
        self.s_nhead = s_nhead  # 空间注意力头的数量
        self.beta = beta  # 门控机制的温度参数
        
        self.model = MASTER(
            d_feat=self.d_feat,
            d_model=self.d_model,
            t_nhead=self.t_nhead,
            s_nhead=self.s_nhead,
            T_dropout=self.T_dropout_rate,
            S_dropout=self.S_dropout_rate,
            gate_input_start_index=self.gate_input_start_index,
            gate_input_end_index=self.gate_input_end_index,
            beta=self.beta,
            time_scales=[1, 3, 5, 10]
        )
        # 如果父类有init_model方法，调用它；否则，可以去掉
        if hasattr(super(), 'init_model'):
            super().init_model()
        self.loss_fn = nn.MSELoss()

    def get_loss(self, pred, target):
        return self.loss_fn(pred, target)