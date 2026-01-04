import torch
from torch import nn
from torch.nn.modules.linear import Linear
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.normalization import LayerNorm
import math

from base_model import SequenceModel  # 从base_model模块中导入SequenceModel类


# 位置编码类，用于为输入序列添加位置信息
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        # 初始化位置编码矩阵，大小为(max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        # 生成位置索引，从0到max_len-1
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 计算div_term，用于调整正弦和余弦函数的频率
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # 对偶数位置使用正弦函数编码
        pe[:, 0::2] = torch.sin(position * div_term)
        # 对奇数位置使用余弦函数编码
        pe[:, 1::2] = torch.cos(position * div_term)
        # 将位置编码矩阵注册为模型的缓冲区，不参与梯度更新
        self.register_buffer("pe", pe)

    def forward(self, x):
        # 将位置编码加到输入x上，x的形状为(batch_size, seq_len, d_model)
        return x + self.pe[:x.shape[1], :]


# 自注意力机制类，用于处理序列内的注意力计算
class SAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout):
        super().__init__()

        self.d_model = d_model  # 模型的维度
        self.nhead = nhead  # 注意力头的数量
        self.temperature = math.sqrt(self.d_model / nhead)  # 温度参数，用于缩放注意力分数

        # 定义线性变换层，用于生成查询(Q)、键(K)、值(V)
        self.qtrans = nn.Linear(d_model, d_model, bias=False)
        self.ktrans = nn.Linear(d_model, d_model, bias=False)
        self.vtrans = nn.Linear(d_model, d_model, bias=False)

        # 为每个注意力头创建一个Dropout层
        attn_dropout_layer = []
        for i in range(nhead):
            attn_dropout_layer.append(Dropout(p=dropout))
        self.attn_dropout = nn.ModuleList(attn_dropout_layer)

        # 输入层的LayerNorm
        self.norm1 = LayerNorm(d_model, eps=1e-5)

        # FFN层的LayerNorm
        self.norm2 = LayerNorm(d_model, eps=1e-5)
        # 定义前馈神经网络(FFN)
        self.ffn = nn.Sequential(
            Linear(d_model, d_model),
            nn.ReLU(),
            Dropout(p=dropout),
            Linear(d_model, d_model),
            Dropout(p=dropout)
        )

    def forward(self, x):
        # 对输入进行LayerNorm
        x = self.norm1(x)
        # 生成查询(Q)、键(K)、值(V)，并调整维度顺序
        q = self.qtrans(x).transpose(0, 1)
        k = self.ktrans(x).transpose(0, 1)
        v = self.vtrans(x).transpose(0, 1)

        dim = int(self.d_model / self.nhead)  # 每个注意力头的维度
        att_output = []
        # 对每个注意力头进行计算
        for i in range(self.nhead):
            if i == self.nhead - 1:
                # 最后一个注意力头处理剩余的维度
                qh = q[:, :, i * dim:]
                kh = k[:, :, i * dim:]
                vh = v[:, :, i * dim:]
            else:
                # 其他注意力头处理对应的维度
                qh = q[:, :, i * dim:(i + 1) * dim]
                kh = k[:, :, i * dim:(i + 1) * dim]
                vh = v[:, :, i * dim:(i + 1) * dim]

            # 计算注意力分数并进行softmax归一化
            atten_ave_matrixh = torch.softmax(torch.matmul(qh, kh.transpose(1, 2)) / self.temperature, dim=-1)
            if self.attn_dropout:
                # 对注意力分数进行Dropout
                atten_ave_matrixh = self.attn_dropout[i](atten_ave_matrixh)
            # 计算注意力输出并调整维度顺序
            att_output.append(torch.matmul(atten_ave_matrixh, vh).transpose(0, 1))
        # 将所有注意力头的输出拼接在一起
        att_output = torch.concat(att_output, dim=-1)

        # 残差连接和LayerNorm
        xt = x + att_output
        xt = self.norm2(xt)
        # 通过FFN并再次进行残差连接
        att_output = xt + self.ffn(xt)

        return att_output


# 时间注意力机制类，用于处理时间序列的注意力计算
class TAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout):
        super().__init__()
        self.d_model = d_model  # 模型的维度
        self.nhead = nhead  # 注意力头的数量
        # 定义线性变换层，用于生成查询(Q)、键(K)、值(V)
        self.qtrans = nn.Linear(d_model, d_model, bias=False)
        self.ktrans = nn.Linear(d_model, d_model, bias=False)
        self.vtrans = nn.Linear(d_model, d_model, bias=False)

        # 为每个注意力头创建一个Dropout层
        self.attn_dropout = []
        if dropout > 0:
            for i in range(nhead):
                self.attn_dropout.append(Dropout(p=dropout))
            self.attn_dropout = nn.ModuleList(self.attn_dropout)

        # 输入层的LayerNorm
        self.norm1 = LayerNorm(d_model, eps=1e-5)
        # FFN层的LayerNorm
        self.norm2 = LayerNorm(d_model, eps=1e-5)
        # 定义前馈神经网络(FFN)
        self.ffn = nn.Sequential(
            Linear(d_model, d_model),
            nn.ReLU(),
            Dropout(p=dropout),
            Linear(d_model, d_model),
            Dropout(p=dropout)
        )

    def forward(self, x):
        # print(x.shape)
        # 对输入进行LayerNorm
        x = self.norm1(x)
        # 生成查询(Q)、键(K)、值(V)
        q = self.qtrans(x)
        k = self.ktrans(x)
        v = self.vtrans(x)

        dim = int(self.d_model / self.nhead)  # 每个注意力头的维度
        att_output = []
        # 对每个注意力头进行计算
        for i in range(self.nhead):
            if i == self.nhead - 1:
                # 最后一个注意力头处理剩余的维度
                qh = q[:, :, i * dim:]
                kh = k[:, :, i * dim:]
                vh = v[:, :, i * dim:]
            else:
                # 其他注意力头处理对应的维度
                qh = q[:, :, i * dim:(i + 1) * dim]
                kh = k[:, :, i * dim:(i + 1) * dim]
                vh = v[:, :, i * dim:(i + 1) * dim]
            # 计算注意力分数并进行softmax归一化
            atten_ave_matrixh = torch.softmax(torch.matmul(qh, kh.transpose(1, 2)), dim=-1)
            if self.attn_dropout:
                # 对注意力分数进行Dropout
                atten_ave_matrixh = self.attn_dropout[i](atten_ave_matrixh)
            # 计算注意力输出
            att_output.append(torch.matmul(atten_ave_matrixh, vh))
        # 将所有注意力头的输出拼接在一起
        att_output = torch.concat(att_output, dim=-1)

        # 残差连接和LayerNorm
        xt = x + att_output
        xt = self.norm2(xt)
        # 通过FFN并再次进行残差连接
        att_output = xt + self.ffn(xt)
        # print(att_output)
        # print(att_output.shape)

        return att_output


# 门控机制类，用于控制特征的权重
class Gate(nn.Module):
    def __init__(self, d_input, d_output, beta=1.0):
        super().__init__()
        # 定义线性变换层
        self.trans = nn.Linear(d_input, d_output)
        self.d_output = d_output  # 输出维度
        self.t = beta  # 温度参数，用于控制softmax的平滑度

    def forward(self, gate_input):
        # 对输入进行线性变换并通过softmax归一化
        output = self.trans(gate_input)
        output = torch.softmax(output / self.t, dim=-1)
        return self.d_output * output  # 返回加权后的输出


# 时间注意力机制类，用于处理时间序列的注意力计算
class TemporalAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # 定义线性变换层
        self.trans = nn.Linear(d_model, d_model, bias=False)

    def forward(self, z):
        # 对输入进行线性变换
        h = self.trans(z)  # [N, T, D]
        # 提取最后一个时间步的查询向量
        query = h[:, -1, :].unsqueeze(-1)
        # 计算注意力分数
        lam = torch.matmul(h, query).squeeze(-1)  # [N, T, D] --> [N, T]
        lam = torch.softmax(lam, dim=1).unsqueeze(1)  # 对注意力分数进行softmax归一化
        # 计算加权输出
        output = torch.matmul(lam, z).squeeze(1)  # [N, 1, T], [N, T, D] --> [N, 1, D]
        return output


# MASTER模型类，整合了多个注意力机制和门控机制
class MASTER(nn.Module):
    def __init__(self, d_feat, d_model, t_nhead, s_nhead, T_dropout_rate, S_dropout_rate, gate_input_start_index, gate_input_end_index, beta):
        super(MASTER, self).__init__()
        # 市场特征的门控机制
        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index
        self.d_gate_input = (gate_input_end_index - gate_input_start_index)  # 门控输入的维度
        self.feature_gate = Gate(self.d_gate_input, d_feat, beta=beta)

        # 定义模型的各个层
        self.layers = nn.Sequential(
            # 特征层
            nn.Linear(d_feat, d_model),
            PositionalEncoding(d_model),
            # 股票内的时间注意力机制
            TAttention(d_model=d_model, nhead=t_nhead, dropout=T_dropout_rate),
            # 股票间的空间注意力机制
            SAttention(d_model=d_model, nhead=s_nhead, dropout=S_dropout_rate),
            # 时间注意力机制
            TemporalAttention(d_model=d_model),
            # 解码器
            nn.Linear(d_model, 1)
        )

    def forward(self, x):
        # 提取输入的特征部分
        src = x[:, :, :self.gate_input_start_index]  # N, T, D
        # 提取门控输入
        gate_input = x[:, -1, self.gate_input_start_index:self.gate_input_end_index]
        # 对特征进行门控加权
        src = src * torch.unsqueeze(self.feature_gate(gate_input), dim=1)
       
        # 通过模型的各个层并输出结果
        output = self.layers(src).squeeze(-1)

        return output


# MASTERModel类，继承自SequenceModel，用于初始化和管理MASTER模型
class MASTERModel(SequenceModel):
    def __init__(
            self, d_feat, d_model, t_nhead, s_nhead, gate_input_start_index, gate_input_end_index,
            T_dropout_rate, S_dropout_rate, beta, **kwargs,
    ):
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

        self.init_model()

    def init_model(self):
        # 初始化MASTER模型
        self.model = MASTER(d_feat=self.d_feat, d_model=self.d_model, t_nhead=self.t_nhead, s_nhead=self.s_nhead,
                                   T_dropout_rate=self.T_dropout_rate, S_dropout_rate=self.S_dropout_rate,
                                   gate_input_start_index=self.gate_input_start_index,
                                   gate_input_end_index=self.gate_input_end_index, beta=self.beta)
        super(MASTERModel, self).init_model()
