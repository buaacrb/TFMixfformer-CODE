import numpy as np
import torch
import torch.nn as nn


class DynamicMixAttentionFusion(nn.Module):
    def __init__(self, seq_len=96):
        super(DynamicMixAttentionFusion, self).__init__()

        
        self.time_weights_en1 = nn.Parameter(torch.rand(seq_len))
        self.freq_weights_en1 = nn.Parameter(torch.rand(seq_len))

        self.time_weights = self.time_weights_en1
        self.freq_weights = self.freq_weights_en1

    def forward(self, time_features, freq_features):

        total_weight = self.time_weights + self.freq_weights
        total_weight = total_weight + (total_weight == 0).float() * 1e-8  # 防止除零
        
        norm_time_weights = self.time_weights / total_weight
        norm_freq_weights = self.freq_weights / total_weight

        norm_time_weights = norm_time_weights.unsqueeze(-1)
        norm_freq_weights = norm_freq_weights.unsqueeze(-1)

        weighted_time = time_features * norm_time_weights
        weighted_freq = freq_features * norm_freq_weights

        fused_features = weighted_time + weighted_freq

        return fused_features



def get_frequency_modes(seq_len, modes=64, mode_select_method='random'):
    """
    get modes on frequency domain:
    'random' means sampling randomly;
    'else' means sampling the lowest modes;
    """
    modes = min(modes, seq_len//2)
    if mode_select_method == 'random':
        index = list(range(0, seq_len // 2))
        np.random.shuffle(index)
        index = index[:modes]
    else:
        index = list(range(0, modes))
    index.sort()
    return index


def get_autocorr(x, max_lag):
    B,H,E,L = x.shape
    autocorr_x = torch.zeros_like(x)

    x_centered = x - x.mean(dim=-1, keepdim=True)
    denominator = torch.sum(x_centered ** 2, dim=-1, keepdim=True)

    for lag in range(max_lag):
        shifted = torch.cat([x_centered[..., lag:], x_centered[..., :lag]], dim=-1)
        numerator = torch.sum(x_centered * shifted, dim=-1, keepdim=True)
        autocorr_x[..., lag] = (numerator / denominator).squeeze(-1)

    return autocorr_x


# ########## fourier layer #############
class MixSelfAttention(nn.Module):
    def __init__(self, in_channels, out_channels, seq_len, modes=0, mode_select_method='random'):
        super(MixSelfAttention, self).__init__()
        # get modes on frequency domain
        self.index = get_frequency_modes(seq_len, modes=modes, mode_select_method=mode_select_method)

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(8, in_channels // 8, out_channels // 8, len(self.index), dtype=torch.cfloat))
        self.dropout = nn.Dropout(0.1)
        self.fusion = DynamicMixAttentionFusion(seq_len)

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bhi,hio->bho", input, weights)

    def forward(self, tf_queries, queries, keys, values, mask):
        # size = [B, L, H, E]
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        x = queries.permute(0, 2, 3, 1)
        q = x
        scale = self.scale or 1. / sqrt(E)
        
        # Mixe time-domain and time-frequency coefficients
        tf_q = tf_queries.permute(0, 2, 3, 1)
        corr_x = get_autocorr(q, L)
        scores = torch.einsum("bhel,bhes->bhls", corr_x, corr_x)
        scores_tf = torch.einsum("bhel,bhes->bhls", tf_q, tf_q)

        mix_scores = self.fusion(scores, scores_tf)

        A_t = self.dropout(torch.softmax(scale * mix_scores, dim=-1))
        
        # Compute Fourier coefficients
        x_ft = torch.fft.rfft(x, dim=-1)
        # Perform Fourier neural operations
        out_ft = torch.zeros(B, H, E, L // 2 + 1, device=x.device, dtype=torch.cfloat)
        for wi, i in enumerate(self.index):
            out_ft[:, :, :, wi] = self.compl_mul1d(x_ft[:, :, :, i], self.weights1[:, :, :, wi])
        # Return to time domain
        V = torch.einsum("bhls,bshd->blhd", A_t, values).permute(0, 2, 3, 1)
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        res = (V + x)/2
        
        
        return (res, None)


# ########## Fourier Cross Former ####################
class MixCrossAttention(nn.Module):
    def __init__(self, in_channels, out_channels, seq_len_q, seq_len_kv, modes=64, mode_select_method='random',
                 activation='tanh', policy=0):
        super(MixCrossAttention, self).__init__()

        self.activation = activation
        self.in_channels = in_channels
        self.out_channels = out_channels
        # get modes for queries and keys (& values) on frequency domain
        self.index_q = get_frequency_modes(seq_len_q, modes=modes, mode_select_method=mode_select_method)
        self.index_kv = get_frequency_modes(seq_len_kv, modes=modes, mode_select_method=mode_select_method)
        self.fusion = DynamicMixAttentionFusion(seq_len_q)

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(8, in_channels // 8, out_channels // 8, len(self.index_q), dtype=torch.cfloat))
        self.dropout = nn.Dropout(0.1)

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bhi,hio->bho", input, weights)

    def forward(self, tf_queries, queries, keys, values, mask):
        # size = [B, L, H, E]
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)
        
        tf_q = tf_queries.permute(0, 2, 3, 1)
        xq = queries.permute(0, 2, 3, 1)  # size = [B, H, E, L]
        xk = keys.permute(0, 2, 3, 1)
        xv = values.permute(0, 2, 3, 1)

        # Mixe time-domain and time-frequency coefficients
        corr_x = get_autocorr(xq, L)
        
        scores = torch.einsum("bhel,bhes->bhls", corr_x, xk)
        scores_tf = torch.einsum("bhel,bhes->bhls", tf_q, xk)

        mix_scores = self.fusion(scores, scores_tf)
        
        A_t = self.dropout(torch.softmax(scale * mix_scores, dim=-1))

        xq_ft_ = torch.zeros(B, H, E, len(self.index_q), device=xq.device, dtype=torch.cfloat)
        xq_ft = torch.fft.rfft(xq, dim=-1)
        for i, j in enumerate(self.index_q):
            xq_ft_[:, :, :, i] = xq_ft[:, :, :, j]
        xk_ft_ = torch.zeros(B, H, E, len(self.index_kv), device=xq.device, dtype=torch.cfloat)
        xk_ft = torch.fft.rfft(xk, dim=-1)
        for i, j in enumerate(self.index_kv):
            xk_ft_[:, :, :, i] = xk_ft[:, :, :, j]

        # perform attention mechanism on frequency domain
        xqk_ft = (torch.einsum("bhex,bhey->bhxy", xq_ft_, xk_ft_))
        if self.activation == 'tanh':
            xqk_ft = xqk_ft.tanh()
        elif self.activation == 'softmax':
            xqk_ft = torch.softmax(abs(xqk_ft), dim=-1)
            xqk_ft = torch.complex(xqk_ft, torch.zeros_like(xqk_ft))
        else:
            raise Exception('{} actiation function is not implemented'.format(self.activation))
        xqkv_ft = torch.einsum("bhxy,bhey->bhex", xqk_ft, xk_ft_)
        xqkvw = torch.einsum("bhex,heox->bhox", xqkv_ft, self.weights1)
        out_ft = torch.zeros(B, H, E, L // 2 + 1, device=xq.device, dtype=torch.cfloat)
        for i, j in enumerate(self.index_q):
            out_ft[:, :, :, j] = xqkvw[:, :, :, i]
        # Return to time domain
        out = torch.fft.irfft(out_ft / self.in_channels / self.out_channels, n=xq.size(-1))
        
        V = torch.einsum("bhls,bshd->blhd", A_t, values).permute(0, 2, 3, 1)
        
        res = (V + out)/2
        
        return (res, None)
