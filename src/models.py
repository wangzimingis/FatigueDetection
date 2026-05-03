# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat
import math
from copy import deepcopy


class ChannelAttention1D(nn.Module):
    """一维通道注意力模块 (适用于时序数据)"""

    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.in_channels = in_channels

        # 平均池化和最大池化
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        # 注意力网络
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (batch, channels, length)
        batch_size, channels, length = x.shape

        # 平均池化和最大池化
        avg_out = self.fc(self.avg_pool(x).view(batch_size, -1))
        max_out = self.fc(self.max_pool(x).view(batch_size, -1))

        # 合并注意力权重
        attention = self.sigmoid(avg_out + max_out).view(batch_size, channels, 1)

        # 应用注意力
        return x * attention


class TemporalAttention(nn.Module):
    """时序注意力模块"""

    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        # x: (batch, seq_len, hidden_dim)
        attention_weights = self.attention(x)  # (batch, seq_len, 1)
        attention_weights = F.softmax(attention_weights, dim=1)
        weighted_x = torch.sum(x * attention_weights, dim=1)
        return weighted_x, attention_weights.squeeze(-1)


class MultiHeadSelfAttention(nn.Module):
    """多头自注意力模块"""

    def __init__(self, d_model, num_heads=8, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model必须能被num_head整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 线性变换层
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        # 线性变换并分头
        q = self.q_linear(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        # 应用注意力
        out = torch.matmul(attention, v)

        # 合并多头
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        out = self.out(out)

        return out, attention


class EEGCNN(nn.Module):
    """EEG CNN特征提取器"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        # 根据特征类型确定特征维度
        if config.feature_type == '2Hz':
            feature_dim = config.frequency_bands
        else:
            feature_dim = config.five_bands

        # 第一层卷积：处理空间信息
        self.conv1 = nn.Conv2d(1, config.cnn_channels[0],
                               kernel_size=(3, 3),
                               padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(config.cnn_channels[0])

        # 第二层卷积
        self.conv2 = nn.Conv2d(config.cnn_channels[0], config.cnn_channels[1],
                               kernel_size=(3, 3),
                               padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(config.cnn_channels[1])

        # 第三层卷积
        self.conv3 = nn.Conv2d(config.cnn_channels[1], config.cnn_channels[2],
                               kernel_size=(3, 3),
                               padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(config.cnn_channels[2])

        # 池化层
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout2d(config.dropout_rate)

        # 自适应池化
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 通道注意力
        self.channel_attention = ChannelAttention1D(config.cnn_channels[2])

        # 计算输出维度
        self._calculate_output_dim(feature_dim)

    def _calculate_output_dim(self, feature_dim):
        """计算输出维度"""
        # 模拟输入以计算输出维度
        dummy_input = torch.randn(1, 1, self.config.eeg_channels, feature_dim)

        x = F.relu(self.bn1(self.conv1(dummy_input)))
        x = self.pool(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.adaptive_pool(x)

        self.output_dim = x.numel()

    def forward(self, x):
        # x: (batch, channels, features)
        # 添加通道维度
        x = x.unsqueeze(1)  # (batch, 1, channels, features)

        # 第一层卷积
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.dropout(x)

        # 第二层卷积
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout(x)

        # 第三层卷积
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.adaptive_pool(x)

        # 调整维度以应用通道注意力
        x = x.squeeze(-1).squeeze(-1)  # (batch, channels)

        # 通道注意力
        x = self.channel_attention(x.unsqueeze(-1)).squeeze(-1)

        return x


class EEGLSTM(nn.Module):
    """EEG LSTM特征提取器"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        # 根据特征类型确定特征维度
        if config.feature_type == '2Hz':
            feature_dim = config.frequency_bands
        else:
            feature_dim = config.five_bands

        # 输入投影层
        self.input_proj = nn.Linear(feature_dim, 64)

        # BiLSTM层
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=config.lstm_hidden,
            num_layers=config.lstm_layers,
            bidirectional=config.lstm_bidirectional,
            batch_first=True,
            dropout=config.dropout_rate if config.lstm_layers > 1 else 0
        )

        # 注意力机制
        lstm_output_dim = config.lstm_hidden * 2 if config.lstm_bidirectional else config.lstm_hidden
        self.temporal_attention = TemporalAttention(lstm_output_dim)

        # Dropout
        self.dropout = nn.Dropout(config.dropout_rate)

        # 输出维度
        self.output_dim = lstm_output_dim

    def forward(self, x):
        # x: (batch, channels, features)
        # 输入投影：将 features 映射到 64
        x = self.input_proj(x)          # (batch, channels, 64)
        x = F.relu(x)

        # LSTM 处理，输入形状 (batch, seq_len=channels, input_size=64)
        lstm_out, (h_n, c_n) = self.lstm(x)

        # 时序注意力
        attended_features, attention_weights = self.temporal_attention(lstm_out)

        # Dropout
        attended_features = self.dropout(attended_features)

        return attended_features, attention_weights


class EOGFeatureExtractor(nn.Module):
    """EOG特征提取器"""

    def __init__(self, config):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(config.eog_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout_rate),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout_rate),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True)
        )

        self.output_dim = 32

    def forward(self, x):
        return self.network(x)


class PositionalEncoding(nn.Module):
    """位置编码模块"""

    def __init__(self, d_model, max_len=5000):
        super().__init__()

        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return x


class TransformerEncoderLayerNoCausal(nn.Module):
    """Transformer编码器层 - 修复版本，避免is_causal参数问题"""

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()

        # 自注意力机制 - 使用更兼容的版本
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        # 前馈网络
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # 归一化层
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout层
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # 激活函数
        self.activation = nn.ReLU()

    def forward(self, src, src_mask=None, src_key_padding_mask=None, **kwargs):
        """前向传播 - 移除了is_causal参数，使用**kwargs接受额外参数"""
        # 多头自注意力 - 兼容不同版本的PyTorch
        try:
            # 新版本PyTorch可能需要is_causal参数
            src2, _ = self.self_attn(src, src, src,
                                     attn_mask=src_mask,
                                     key_padding_mask=src_key_padding_mask,
                                     need_weights=False)
        except TypeError:
            # 旧版本PyTorch
            src2, _ = self.self_attn(src, src, src,
                                     attn_mask=src_mask,
                                     key_padding_mask=src_key_padding_mask)

        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # 前馈网络
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src

# ==================== 新增 HyperLSTM 核心组件 ====================
class LayerNorm(nn.Module):
    """层归一化（与之前一致）"""
    def __init__(self, num_features, eps=1e-6):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.gamma = nn.Parameter(torch.FloatTensor(num_features))
        self.beta = nn.Parameter(torch.FloatTensor(num_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.gamma, 1)
        nn.init.constant_(self.beta, 0)

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class ParallelLayerNorm(nn.Module):
    """并行层归一化（同时处理多个输入张量）"""
    def __init__(self, num_inputs, num_features, eps=1e-6):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_features = num_features
        self.eps = eps
        self.gamma = nn.Parameter(torch.FloatTensor(num_inputs, num_features))
        self.beta = nn.Parameter(torch.FloatTensor(num_inputs, num_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.gamma, 1)
        nn.init.constant_(self.beta, 0)

    def forward(self, *inputs):
        inputs_stacked = torch.stack(inputs, dim=-2)  # (batch, num_inputs, features)
        mean = inputs_stacked.mean(dim=-1, keepdim=True)
        std = inputs_stacked.std(dim=-1, keepdim=True)
        outputs_stacked = (self.gamma * (inputs_stacked - mean) / (std + self.eps) + self.beta)
        return torch.unbind(outputs_stacked, dim=-2)


class LSTMCell(nn.Module):
    """标准 LSTM Cell（支持 LayerNorm）"""
    def __init__(self, input_size, hidden_size, use_layer_norm, dropout_prob=0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_layer_norm = use_layer_norm
        self.dropout_prob = dropout_prob

        self.linear_ih = nn.Linear(input_size, 4 * hidden_size)
        self.linear_hh = nn.Linear(hidden_size, 4 * hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout_prob)
        if use_layer_norm:
            self.ln_ifgo = ParallelLayerNorm(4, hidden_size)
            self.ln_c = LayerNorm(hidden_size)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear_ih.weight)
        nn.init.constant_(self.linear_ih.bias, 0)
        nn.init.orthogonal_(self.linear_hh.weight)
        if self.use_layer_norm:
            self.ln_ifgo.reset_parameters()
            self.ln_c.reset_parameters()

    def forward(self, x, state):
        if state is None:
            batch_size = x.size(0)
            h = x.new_zeros(batch_size, self.hidden_size)
            c = x.new_zeros(batch_size, self.hidden_size)
            state = (h, c)
        h, c = state
        lstm_vec = self.linear_ih(x) + self.linear_hh(h)
        i, f, g, o = lstm_vec.chunk(4, dim=1)
        if self.use_layer_norm:
            i, f, g, o = self.ln_ifgo(i, f, g, o)
        f = f + 1
        new_c = c * f.sigmoid() + i.sigmoid() * self.dropout(g.tanh())
        if self.use_layer_norm:
            new_c = self.ln_c(new_c)
        new_h = new_c.tanh() * o.sigmoid()
        return new_h, (new_h, new_c)


class HyperLSTMCell(nn.Module):
    """超网络 LSTM Cell（核心）"""
    def __init__(self, input_size, hidden_size,
                 hyper_hidden_size, hyper_embedding_size,
                 use_layer_norm, dropout_prob):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hyper_hidden_size = hyper_hidden_size
        self.hyper_embedding_size = hyper_embedding_size
        self.use_layer_norm = use_layer_norm
        self.dropout_prob = dropout_prob

        # Hyper LSTM cell
        self.hyper_cell = LSTMCell(
            input_size + hidden_size,
            hyper_hidden_size,
            use_layer_norm,
            dropout_prob
        )

        # Projection layers (from hyper hidden state to embedding)
        # suffixes: h, x, b
        for suffix in ['h', 'x', 'b']:
            for name in ['i', 'f', 'g', 'o']:
                setattr(self, f'hyper_proj_{name}{suffix}',
                        nn.Linear(hyper_hidden_size, hyper_embedding_size,
                                  bias=False if suffix == 'b' else True))
        # Scaling layers (from embedding to main LSTM parameters)
        for suffix in ['h', 'x', 'b']:
            for name in ['i', 'f', 'g', 'o']:
                setattr(self, f'hyper_scale_{name}{suffix}',
                        nn.Linear(hyper_embedding_size, hidden_size, bias=False))

        # Main LSTM parameters
        self.linear_ih = nn.Linear(input_size, 4 * hidden_size, bias=False)
        self.linear_hh = nn.Linear(hidden_size, 4 * hidden_size, bias=False)
        self.bias = nn.Parameter(torch.FloatTensor(4 * hidden_size))

        if use_layer_norm:
            self.ln_ifgo = ParallelLayerNorm(4, hidden_size)
            self.ln_c = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.reset_parameters()

    def reset_parameters(self):
        self.hyper_cell.reset_parameters()
        # Projection layers init
        for suffix in ['h', 'x', 'b']:
            for name in ['i', 'f', 'g', 'o']:
                proj = getattr(self, f'hyper_proj_{name}{suffix}')
                if suffix == 'b':
                    nn.init.normal_(proj.weight, 0, 0.01)
                    if proj.bias is not None:
                        nn.init.constant_(proj.bias, 0)
                else:
                    nn.init.constant_(proj.weight, 0)
                    if proj.bias is not None:
                        nn.init.constant_(proj.bias, 1)

        # Scaling layers init
        for suffix in ['h', 'x', 'b']:
            for name in ['i', 'f', 'g', 'o']:
                scale = getattr(self, f'hyper_scale_{name}{suffix}')
                nn.init.constant_(scale.weight, 0.1 / self.hyper_embedding_size)

        # Main LSTM
        nn.init.xavier_uniform_(self.linear_ih.weight)
        nn.init.orthogonal_(self.linear_hh.weight)
        nn.init.constant_(self.bias, 0)

        if self.use_layer_norm:
            self.ln_ifgo.reset_parameters()
            self.ln_c.reset_parameters()

    def _get_hyper_vector(self, hyper_h, fullname):
        proj = getattr(self, f'hyper_proj_{fullname}')
        scale = getattr(self, f'hyper_scale_{fullname}')
        return scale(proj(hyper_h))

    def forward(self, x, state, hyper_state, mask=None):
        if state is None:
            batch_size = x.size(0)
            h = x.new_zeros(batch_size, self.hidden_size)
            c = x.new_zeros(batch_size, self.hidden_size)
            state = (h, c)
        h, c = state

        # Hyper LSTM step
        hyper_input = torch.cat([x, h], dim=1)
        new_hyper_h, new_hyper_state = self.hyper_cell(hyper_input, hyper_state)

        # Compute main LSTM gates
        xh = self.linear_ih(x)
        hh = self.linear_hh(h)
        ix, fx, gx, ox = xh.chunk(4, dim=1)
        ih, fh, gh, oh = hh.chunk(4, dim=1)
        ib, fb, gb, ob = self.bias.chunk(4, dim=0)

        # Apply hyper vectors to each component
        ix = ix * self._get_hyper_vector(new_hyper_h, 'ix')
        fx = fx * self._get_hyper_vector(new_hyper_h, 'fx')
        gx = gx * self._get_hyper_vector(new_hyper_h, 'gx')
        ox = ox * self._get_hyper_vector(new_hyper_h, 'ox')

        ih = ih * self._get_hyper_vector(new_hyper_h, 'ih')
        fh = fh * self._get_hyper_vector(new_hyper_h, 'fh')
        gh = gh * self._get_hyper_vector(new_hyper_h, 'gh')
        oh = oh * self._get_hyper_vector(new_hyper_h, 'oh')

        ib = ib + self._get_hyper_vector(new_hyper_h, 'ib')
        fb = fb + self._get_hyper_vector(new_hyper_h, 'fb')
        gb = gb + self._get_hyper_vector(new_hyper_h, 'gb')
        ob = ob + self._get_hyper_vector(new_hyper_h, 'ob')

        i = ix + ih + ib
        f = fx + fh + fb + 1
        g = gx + gh + gb
        o = ox + oh + ob

        if self.use_layer_norm:
            i, f, g, o = self.ln_ifgo(i, f, g, o)
        new_c = c * f.sigmoid() + i.sigmoid() * self.dropout(g.tanh())
        if self.use_layer_norm:
            new_c = self.ln_c(new_c)
        new_h = new_c.tanh() * o.sigmoid()

        if mask is not None:
            mask = mask.unsqueeze(1)
            new_h = new_h * mask + h * (1 - mask)
            new_c = new_c * mask + c * (1 - mask)

        return new_h, (new_h, new_c), new_hyper_state

class EEGTransformer(nn.Module):
    """EEG Transformer特征提取器 - 修复版本"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        # 根据特征类型确定特征维度
        if config.feature_type == '2Hz':
            feature_dim = config.frequency_bands
        else:
            feature_dim = config.five_bands

        # 输入投影层
        self.input_projection = nn.Linear(feature_dim, config.transformer_dim)

        # 位置编码
        self.pos_encoder = PositionalEncoding(config.transformer_dim)

        # 创建Transformer编码器层
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=config.transformer_dim,
            nhead=config.transformer_heads,
            dim_feedforward=config.transformer_ff_dim,
            dropout=config.dropout_rate,
            batch_first=True,
            activation='relu'
        )

        # 创建Transformer编码器
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers,
            num_layers=config.transformer_layers
        )

        # 分类token
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.transformer_dim))

        # 输出层
        self.output_norm = nn.LayerNorm(config.transformer_dim)
        self.dropout = nn.Dropout(config.dropout_rate)

        self.output_dim = config.transformer_dim

    def forward(self, x):
        # x: (batch, channels, features)
        batch_size = x.size(0)

        # 投影到transformer维度
        x = self.input_projection(x)  # (batch, channels, transformer_dim)

        # 添加分类token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (batch, channels+1, transformer_dim)

        # 位置编码
        x = self.pos_encoder(x)

        # Transformer编码
        x = self.transformer_encoder(x)

        # 取分类token的输出
        x = x[:, 0, :]  # (batch, transformer_dim)

        # 层归一化和dropout
        x = self.output_norm(x)
        x = self.dropout(x)

        return x


class EEGHyperLSTM(nn.Module):
    """EEG HyperLSTM 特征提取器 (超网络LSTM)"""
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 特征维度
        if config.feature_type == '2Hz':
            feature_dim = config.frequency_bands
        else:
            feature_dim = config.five_bands

        # 输入投影层（将特征维度映射到 64）
        self.input_proj = nn.Linear(feature_dim, 64)
        self.relu = nn.ReLU()

        # HyperLSTM 参数
        self.hidden_size = config.lstm_hidden          # 主 LSTM 隐藏维度
        self.hyper_hidden_size = config.hyper_hidden_size    # 超网络隐藏维度
        self.hyper_embedding_size = config.hyper_embedding_size

        self.use_layer_norm = config.use_layer_norm
        dropout = config.dropout_rate

        # 创建 HyperLSTM Cell
        self.hyperlstm_cell = HyperLSTMCell(
            input_size=64,
            hidden_size=self.hidden_size,
            hyper_hidden_size=self.hyper_hidden_size,
            hyper_embedding_size=self.hyper_embedding_size,
            use_layer_norm=self.use_layer_norm,
            dropout_prob=dropout
        )

        # 时序注意力（与 EEGLSTM 保持一致）
        self.temporal_attention = TemporalAttention(self.hidden_size)
        self.dropout = nn.Dropout(config.dropout_rate)

        self.output_dim = self.hidden_size

    def forward(self, x):
        # x: (batch, channels, features)
        batch_size, channels, _ = x.shape

        # 投影特征
        x = self.input_proj(x)          # (batch, channels, 64)
        x = self.relu(x)

        # 将 channels 作为时间步，依次输入 HyperLSTM Cell
        state = None          # (h, c)
        hyper_state = None    # (hyper_h, hyper_c)
        outputs = []

        for t in range(channels):
            x_t = x[:, t, :]   # (batch, 64)
            h_t, state, hyper_state = self.hyperlstm_cell(x_t, state, hyper_state)
            outputs.append(h_t)

        # 堆叠所有时间步的输出 (batch, channels, hidden_size)
        outputs = torch.stack(outputs, dim=1)

        # 时序注意力池化 -> 生成最终固定长度的特征向量
        attended_features, attention_weights = self.temporal_attention(outputs)
        attended_features = self.dropout(attended_features)

        return attended_features, attention_weights

class MultiModalFusion(nn.Module):
    """多模态融合模块"""

    def __init__(self, eeg_dim, eog_dim, config):
        super().__init__()
        self.config = config
        self.fusion_method = config.fusion_method

        if self.fusion_method == 'concatenate':
            # 简单拼接
            self.fusion_layer = nn.Sequential(
                nn.Linear(eeg_dim + eog_dim, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(config.dropout_rate)
            )

        elif self.fusion_method == 'attention':
            # 注意力融合
            self.eeg_projection = nn.Linear(eeg_dim, 256)
            self.eog_projection = nn.Linear(eog_dim, 256)

            self.attention = nn.Sequential(
                nn.Linear(512, 128),
                nn.Tanh(),
                nn.Linear(128, 2)
            )

        elif self.fusion_method == 'cross_attention':
            # 交叉注意力融合
            self.eeg_projection = nn.Linear(eeg_dim, 256)
            self.eog_projection = nn.Linear(eog_dim, 256)

            self.cross_attention = MultiHeadSelfAttention(256, num_heads=8)

            self.fusion_layer = nn.Sequential(
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(config.dropout_rate)
            )

        else:
            raise ValueError(f"未知的融合方法: {self.fusion_method}")

    def forward(self, eeg_features, eog_features):
        if self.fusion_method == 'concatenate':
            # 简单拼接
            fused = torch.cat([eeg_features, eog_features], dim=1)
            fused = self.fusion_layer(fused)

        elif self.fusion_method == 'attention':
            # 注意力加权融合
            eeg_proj = self.eeg_projection(eeg_features)
            eog_proj = self.eog_projection(eog_features)

            combined = torch.cat([eeg_proj, eog_proj], dim=1)
            attention_weights = F.softmax(self.attention(combined), dim=1)

            fused = (attention_weights[:, 0:1] * eeg_proj +
                     attention_weights[:, 1:2] * eog_proj)

        elif self.fusion_method == 'cross_attention':
            # 交叉注意力融合
            eeg_proj = self.eeg_projection(eeg_features).unsqueeze(1)
            eog_proj = self.eog_projection(eog_features).unsqueeze(1)

            # 交叉注意力：用EEG作为query，EOG作为key和value
            attended_eeg, _ = self.cross_attention(eeg_proj, eog_proj, eog_proj)

            fused = attended_eeg.squeeze(1)
            fused = self.fusion_layer(fused)

        return fused


class FatigueCNN(nn.Module):
    """CNN疲劳检测模型（基线模型）"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        # 根据特征类型确定特征维度
        if config.feature_type == '2Hz':
            feature_dim = config.frequency_bands
        else:
            feature_dim = config.five_bands

        # 卷积层
        self.conv_layers = nn.Sequential(
            # 第一层
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(config.dropout_rate),

            # 第二层
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(config.dropout_rate),

            # 第三层
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # 全连接层
        self.fc_layers = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout_rate),

            nn.Linear(64, config.num_classes if config.task_type == 'classification' else 1)
        )

    def forward(self, x):
        # 添加通道维度
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (batch, 1, channels, features)

        # 卷积特征提取
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # 展平

        # 全连接分类
        x = self.fc_layers(x)

        return x


class MultiModalFatigueModel(nn.Module):
    """多模态疲劳检测模型（EEG + EOG + 可选CNN/LSTM/Transformer）"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        # EEG特征提取器
        if config.model_type == 'cnn':
            self.eeg_extractor = EEGCNN(config)
            eeg_output_dim = self.eeg_extractor.output_dim

        elif config.model_type == 'lstm':
            self.eeg_extractor = EEGLSTM(config)
            eeg_output_dim = self.eeg_extractor.output_dim

        elif config.model_type == 'transformer':
            self.eeg_extractor = EEGTransformer(config)
            eeg_output_dim = self.eeg_extractor.output_dim

        elif config.model_type == 'hyperlstm':
            self.eeg_extractor = EEGHyperLSTM(config)
            eeg_output_dim = self.eeg_extractor.output_dim

        else:
            raise ValueError(f"未知的模型类型: {config.model_type}")

        # EOG特征提取器
        if config.use_eog:
            self.eog_extractor = EOGFeatureExtractor(config)
            eog_output_dim = self.eog_extractor.output_dim
        else:
            self.eog_extractor = None
            eog_output_dim = 0

        # 多模态融合
        if config.use_eog and config.use_multimodal and eog_output_dim > 0:
            self.fusion = MultiModalFusion(eeg_output_dim, eog_output_dim, config)
            fusion_output_dim = 256
        else:
            self.fusion = None
            fusion_output_dim = eeg_output_dim

        # 分类/回归头
        self.classifier = nn.Sequential(
            nn.Linear(fusion_output_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout_rate),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout_rate),

            nn.Linear(64, config.num_classes if config.task_type == 'classification' else 1)
        )

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)

    def forward(self, eeg, eog=None):
        # EEG特征提取
        if isinstance(self.eeg_extractor, (EEGLSTM, EEGHyperLSTM)):
            eeg_features, attention_weights = self.eeg_extractor(eeg)
        else:
            eeg_features = self.eeg_extractor(eeg)
            attention_weights = None

        # EOG特征提取
        if self.eog_extractor is not None and eog is not None:
            eog_features = self.eog_extractor(eog)
        else:
            eog_features = None

        # 多模态融合
        if self.fusion is not None and eog_features is not None:
            fused_features = self.fusion(eeg_features, eog_features)
        else:
            fused_features = eeg_features

        # 分类/回归
        output = self.classifier(fused_features)
        return output


# 模型工厂函数
def create_model(config):
    """创建模型工厂函数"""

    if config.model_type in ['cnn', 'lstm', 'transformer', 'hyperlstm']:
        model = MultiModalFatigueModel(config)
        model_name = f"多模态{config.model_type.upper()}模型"

    elif config.model_type == 'multimodal_cnn_lstm':
        # 创建一个结合CNN和LSTM的混合模型
        hybrid_config = deepcopy(config)
        hybrid_config.model_type = 'lstm'  # 使用LSTM作为基础

        model = MultiModalFatigueModel(hybrid_config)
        model_name = "多模态CNN-LSTM混合模型"

    elif config.model_type == 'multimodal_transformer':
        # 使用Transformer作为基础
        trans_config = deepcopy(config)
        trans_config.model_type = 'transformer'

        model = MultiModalFatigueModel(trans_config)
        model_name = "多模态Transformer模型"

    else:
        # 回退到CNN基线模型
        model = FatigueCNN(config)
        model_name = "CNN基线模型"

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"创建模型: {model_name}")
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

    return model


def count_parameters(model):
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# 测试函数
def test_models():
    """测试所有模型"""
    from config import Config

    print("=" * 60)
    print("测试模型")
    print("=" * 60)

    # 创建配置
    config = Config()
    config.eeg_channels = 17
    config.frequency_bands = 25
    config.five_bands = 5
    config.eog_features = 36
    config.cnn_channels = (64, 128, 256)
    config.lstm_hidden = 128
    config.transformer_dim = 256

    # 测试不同模型类型
    model_types = ['cnn', 'lstm', 'transformer', 'multimodal_cnn_lstm', 'multimodal_transformer']

    for model_type in model_types:
        print(f"\n测试模型类型: {model_type}")
        print("-" * 40)

        try:
            # 更新配置
            config.model_type = model_type
            config.use_eog = True
            config.use_multimodal = True
            config.task_type = 'classification'
            config.num_classes = 3

            # 创建模型
            model = create_model(config)

            # 创建模拟输入
            batch_size = 2
            if config.feature_type == '2Hz':
                feature_dim = config.frequency_bands
            else:
                feature_dim = config.five_bands

            eeg_input = torch.randn(batch_size, config.eeg_channels, feature_dim)
            eog_input = torch.randn(batch_size, config.eog_features)

            # 前向传播
            model.eval()
            with torch.no_grad():
                output = model(eeg_input, eog_input)

                print(f"输入形状: EEG={eeg_input.shape}, EOG={eog_input.shape}")
                print(f"输出形状: {output.shape}")

            print("✅ 测试通过")

        except Exception as e:
            print(f"❌ 测试失败: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("所有模型测试完成")
    print("=" * 60)


if __name__ == '__main__':
    test_models()