import torch
from torch import nn
from torch.nn.modules.linear import Linear
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.normalization import LayerNorm
import math


try:
    # 首先尝试相对导入
    from .base_model import SequenceModel
except ImportError:
    # 如果相对导入失败，尝试绝对导入
    from base_model import SequenceModel


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


class SAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.temperature = math.sqrt(self.d_model/nhead)

        self.qtrans = nn.Linear(d_model, d_model, bias=False)
        self.ktrans = nn.Linear(d_model, d_model, bias=False)
        self.vtrans = nn.Linear(d_model, d_model, bias=False)

        # 原实现为逐 head dropout（ModuleList），但计算上没有必要；这里用一次性 Dropout，
        # 同分布、显著减少 Python 循环带来的 kernel 碎片化。
        self.attn_dropout = Dropout(p=dropout) if dropout and dropout > 0 else None

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
        q = self.qtrans(x).transpose(0, 1)   # [T, N, D]
        k = self.ktrans(x).transpose(0, 1)   # [T, N, D]
        v = self.vtrans(x).transpose(0, 1)   # [T, N, D]

        # 向量化 multi-head：一次性 batched matmul，减少 Python 循环与 kernel launch 开销
        T, N, D = q.shape
        if D % self.nhead != 0:
            # 兜底：极少数不整除场景保持旧逻辑（但一般 d_model 可整除 nhead）
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
                atten = torch.softmax(torch.matmul(qh, kh.transpose(1, 2)) / self.temperature, dim=-1)
                if self.attn_dropout is not None:
                    atten = self.attn_dropout(atten)
                att_output.append(torch.matmul(atten, vh).transpose(0, 1))
            att_output = torch.concat(att_output, dim=-1)
        else:
            hd = D // self.nhead
            # [T, N, D] -> [T, H, N, hd]
            qh = q.reshape(T, N, self.nhead, hd).permute(0, 2, 1, 3)
            kh = k.reshape(T, N, self.nhead, hd).permute(0, 2, 1, 3)
            vh = v.reshape(T, N, self.nhead, hd).permute(0, 2, 1, 3)

            atten = torch.matmul(qh, kh.transpose(-2, -1)) / self.temperature   # [T, H, N, N]
            atten = torch.softmax(atten, dim=-1)
            if self.attn_dropout is not None:
                atten = self.attn_dropout(atten)
            out = torch.matmul(atten, vh)  # [T, H, N, hd]
            # [T, H, N, hd] -> [N, T, D]
            att_output = out.permute(0, 2, 1, 3).contiguous().reshape(T, N, D).transpose(0, 1)

        # FFN
        xt = x + att_output
        xt = self.norm2(xt)
        att_output = xt + self.ffn(xt)

        return att_output


class TAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.qtrans = nn.Linear(d_model, d_model, bias=False)
        self.ktrans = nn.Linear(d_model, d_model, bias=False)
        self.vtrans = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout = Dropout(p=dropout) if dropout and dropout > 0 else None

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

        N, T, D = q.shape
        if D % self.nhead != 0:
            # 兜底：保持旧逻辑
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
                atten = torch.softmax(torch.matmul(qh, kh.transpose(1, 2)), dim=-1)
                if self.attn_dropout is not None:
                    atten = self.attn_dropout(atten)
                att_output.append(torch.matmul(atten, vh))
            att_output = torch.concat(att_output, dim=-1)
        else:
            hd = D // self.nhead
            # [N, T, D] -> [N, H, T, hd]
            qh = q.reshape(N, T, self.nhead, hd).permute(0, 2, 1, 3)
            kh = k.reshape(N, T, self.nhead, hd).permute(0, 2, 1, 3)
            vh = v.reshape(N, T, self.nhead, hd).permute(0, 2, 1, 3)

            # 注意：原实现未做 1/sqrt(d) 缩放，这里保持一致
            atten = torch.matmul(qh, kh.transpose(-2, -1))   # [N, H, T, T]
            atten = torch.softmax(atten, dim=-1)
            if self.attn_dropout is not None:
                atten = self.attn_dropout(atten)
            out = torch.matmul(atten, vh)  # [N, H, T, hd]
            att_output = out.permute(0, 2, 1, 3).contiguous().reshape(N, T, D)

        # FFN
        xt = x + att_output
        xt = self.norm2(xt)
        att_output = xt + self.ffn(xt)

        return att_output


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


class MASTER(nn.Module):
    def __init__(self, d_feat, d_model, t_nhead, s_nhead, T_dropout_rate, S_dropout_rate, gate_input_start_index, gate_input_end_index, beta):
        super(MASTER, self).__init__()
        # market
        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index
        self.d_gate_input = (gate_input_end_index - gate_input_start_index) # F'
        self.feature_gate = Gate(self.d_gate_input, d_feat, beta=beta)

        self.encoder = nn.Sequential(
            # feature layer
            nn.Linear(d_feat, d_model),
            PositionalEncoding(d_model),
            # intra-stock aggregation
            TAttention(d_model=d_model, nhead=t_nhead, dropout=T_dropout_rate),
            # inter-stock aggregation
            SAttention(d_model=d_model, nhead=s_nhead, dropout=S_dropout_rate),
            TemporalAttention(d_model=d_model),
        )
        self.decoder = nn.Linear(d_model, 1)

    def encode(self, x):
        src = x[:, :, :self.gate_input_start_index] # N, T, D
        gate_input = x[:, -1, self.gate_input_start_index:self.gate_input_end_index]
        src = src * torch.unsqueeze(self.feature_gate(gate_input), dim=1)
        return self.encoder(src)

    def forward(self, x):
        enc = self.encode(x)
        output = self.decoder(enc)
        output = output.squeeze(-1)
        return output


class MASTERModel(SequenceModel):
    def __init__(
            self, d_feat, d_model, t_nhead, s_nhead, gate_input_start_index, gate_input_end_index,
            T_dropout_rate, S_dropout_rate, beta, **kwargs,
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

        self.init_model()

    def init_model(self):
        self.model = MASTER(d_feat=self.d_feat, d_model=self.d_model, t_nhead=self.t_nhead, s_nhead=self.s_nhead,
                                   T_dropout_rate=self.T_dropout_rate, S_dropout_rate=self.S_dropout_rate,
                                   gate_input_start_index=self.gate_input_start_index,
                                   gate_input_end_index=self.gate_input_end_index, beta=self.beta)
        super(MASTERModel, self).init_model()

    # 256特征值生成输出
    def encode(self, dl_test=None):

        if dl_test is not None:
            encode_test = super().encode(dl_test)

        return (
            encode_test if dl_test is not None else None
        )