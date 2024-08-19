
import numpy as np
import torch
import torch.nn as nn



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

def get_sig_scores(n, queries, keys, scale, corr=0):
    B, L_Q, H, E = queries.shape
    _, L_K, _, D = keys.shape

    q = queries.permute(0, 2, 1, 3) # [B, H, L, E]
    k = keys.permute(0, 2, 1, 3) # [B, H, L, E]

    if corr:
        q_fft = torch.fft.rfft(queries.permute(0, 2, 3, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(keys.permute(0, 2, 3, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        autocorr = torch.fft.irfft(res, dim=-1).permute(0, 1, 3, 2)
    else:
        autocorr = q
    
    amplitudes = autocorr.norm(p=2, dim=-1)  # [B, H, L_Q]

    top_idx = torch.topk(amplitudes, n, dim=2)[1]  # [B, H, n_top]

    queries_selected = torch.gather(q, 2, top_idx.unsqueeze(-1).expand(-1, -1, -1, E))  # [B, H, n_top, E]

    scores = torch.einsum('bhnd,bhmd->bhnm', queries_selected, k) * scale

    full_scores = torch.full((B, H, L_Q, L_K), float('-inf'), device=queries.device, dtype=queries.dtype)

    full_scores.scatter_(2, top_idx.unsqueeze(-1).expand(-1, -1, -1, L_K), scores)

    attn = torch.softmax(full_scores.transpose(2,3), dim=-1)

    return attn


class MixSelfAttention(nn.Module):
    def __init__(self, in_channels, out_channels, seq_len, factor=5, modes=0, mode_select_method='random'):
        super(MixSelfAttention, self).__init__()

        # get modes on frequency domain
        self.factor = factor
        self.index = get_frequency_modes(seq_len, modes=modes, mode_select_method=mode_select_method)

        self.scale = (1 / np.sqrt(out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(8, in_channels // 8, out_channels // 8, len(self.index), dtype=torch.cfloat))
        self.dropout = nn.Dropout(0.1)
        self.weights = nn.Parameter(torch.rand(8, seq_len, 2))


    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bhi,hio->bho", input, weights)

    def forward(self, tf_queries, queries, keys, values, mask):
        # size = [B, L, H, E]
        B, L_Q, H, E = queries.shape
        # _, L_TFQ, _, E = tf_queries.shape
        _, L_K, _, _ = keys.shape
        _, S, _, D = values.shape
        x = queries.permute(0, 2, 3, 1)
        
        n_top = int(self.factor * np.ceil(np.log(L_Q)))
        n_top = min(n_top, L_Q)

        attn_t = get_sig_scores(n_top, queries, keys, self.scale, corr=1)
        attn_tf = get_sig_scores(n_top, tf_queries, tf_queries, self.scale, corr=0)


        # concatenated_scores = torch.cat([full_scores_t, full_scores_tf], dim=-1)
        # transformed_scores = self.transform(concatenated_scores)
        normalized_weights = torch.softmax(self.weights, dim=-1)
        fused_attention = (normalized_weights[:, :, 0].unsqueeze(0).unsqueeze(-1) * attn_t +
                           normalized_weights[:, :, 1].unsqueeze(0).unsqueeze(-1) * attn_tf)
   
        final_attn = self.dropout(fused_attention)

        # Compute Fourier coefficients
        x_ft = torch.fft.rfft(x, dim=-1)
        # Perform Fourier neural operations
        out_ft = torch.zeros(B, H, E, L_Q // 2 + 1, device=x.device, dtype=torch.cfloat)
        for wi, i in enumerate(self.index):
            out_ft[:, :, :, wi] = self.compl_mul1d(x_ft[:, :, :, i], self.weights1[:, :, :, wi])
        # Return to time domain
        x = torch.fft.irfft(out_ft, n=x.size(-1)) #.permute(0, 1, 3, 2)

        # 计算输出
        output = torch.einsum('bhlm,bhmd->bhld', final_attn, values.permute(0, 2, 1, 3)).permute(0, 1, 3, 2) #.permute(0, 2, 1, 3)

        final_output = (output + x)/2

        return (output, None)


class MixCrossAttention(nn.Module):
    def __init__(self, in_channels, out_channels, seq_len_q, seq_len_kv, factor=5, modes=0, mode_select_method='random'):
        super(MixCrossAttention, self).__init__()

        self.factor = factor
        self.activation = 'tanh'
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.index_q = get_frequency_modes(seq_len_q, modes=modes, mode_select_method=mode_select_method)
        self.index_kv = get_frequency_modes(seq_len_kv, modes=modes, mode_select_method=mode_select_method)

        self.scale = (1 / np.sqrt(out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(8, in_channels // 8, out_channels // 8, len(self.index_q), dtype=torch.cfloat))
        self.dropout = nn.Dropout(0.1)
        self.weights = nn.Parameter(torch.rand(8, seq_len_q, 2))


    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bhi,hio->bho", input, weights)

    def forward(self, tf_queries, queries, keys, values, mask):
        # size = [B, L, H, E]
        B, L_Q, H, E = queries.shape
        # _, L_TFQ, _, E = tf_queries.shape
        # _, L_K, _, _ = keys.shape
        _, S, _, D = keys.shape
        x = queries.permute(0, 2, 3, 1)

        xq = queries.permute(0, 2, 3, 1)  # size = [B, H, E, L]
        xk = keys.permute(0, 2, 3, 1)
        xv = values.permute(0, 2, 3, 1)

        if L_Q > S:
            zeros = torch.zeros_like(queries[:, :(L_Q - S), :]).float()
            keys = torch.cat([keys, zeros], dim=1)
            values = torch.cat([values, zeros], dim=1)
        else:
            keys = keys[:, :L_Q, :, :]
            values = values[:, :L_Q, :, :]
        
        n_top = int(self.factor * np.ceil(np.log(L_Q)))
        n_top = min(n_top, L_Q)

        attn_t = get_sig_scores(n_top, queries, keys, self.scale, corr=1)
        attn_tf = get_sig_scores(n_top, tf_queries, keys, self.scale, corr=0)

        normalized_weights = torch.softmax(self.weights, dim=-1)
        fused_attention = (normalized_weights[:, :, 0].unsqueeze(0).unsqueeze(-1) * attn_t +
                           normalized_weights[:, :, 1].unsqueeze(0).unsqueeze(-1) * attn_tf)

        # 应用softmax
        # final_attn = torch.softmax(transformed_scores, dim=-1)
        final_attn = self.dropout(fused_attention)

        # Compute Fourier coefficients
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
        out_ft = torch.zeros(B, H, E, L_Q // 2 + 1, device=xq.device, dtype=torch.cfloat)
        for i, j in enumerate(self.index_q):
            out_ft[:, :, :, j] = xqkvw[:, :, :, i]
        # Return to time domain
        out = torch.fft.irfft(out_ft / self.in_channels / self.out_channels, n=xq.size(-1)) #.permute(0, 1, 3, 2)

        # 计算输出
        output = torch.einsum('bhlm,bhmd->bhld', final_attn, values.permute(0, 2, 1, 3)).permute(0, 1, 3, 2)#.permute(0, 2, 1, 3)
        final_output = (output + out)/2

        return (final_output, None)
