
import numpy as np
import torch
import torch.nn as nn



def get_sig_scores(n, queries, keys, scale, corr=0):
    B, L_Q, H, E = queries.shape
    _, L_K, _, D = keys.shape

    q = queries.permute(0, 2, 1, 3) # [B, H, L, E]
    k = keys.permute(0, 2, 1, 3) # [B, H, L, E]

    #自相关性计算
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

    attn = full_scores.permute(0, 1, 3, 2)

    return attn


class MixSelfAttention(nn.Module):
    def __init__(self, in_channels, out_channels, seq_len, factor=5, modes=0, mode_select_method='random'):
        super(MixSelfAttention, self).__init__()

        self.factor = factor

        self.transform = nn.Linear(2 * seq_len, seq_len)
        self.scale = (1 / np.sqrt(out_channels))

        self.dropout = nn.Dropout(0.1)
        self.weights = nn.Parameter(torch.rand(8, seq_len, 2))


    
    def forward(self, tf_queries, queries, keys, values, mask):
        # size = [B, L, H, E]
        B, L_Q, H, E = queries.shape
        # _, L_TFQ, _, E = tf_queries.shape
        _, L_K, _, _ = keys.shape
        _, S, _, D = values.shape
        
        n_top = int(self.factor * np.ceil(np.log(L_Q)))
        n_top = min(n_top, L_Q)

        attn_t = get_sig_scores(n_top, queries, keys, self.scale, corr=1)
        attn_tf = get_sig_scores(n_top, tf_queries, tf_queries, self.scale, corr=0)

        #拼接两种分数并进行变换处理
        concatenated_scores = torch.cat([attn_t, attn_tf], dim=-1)
        concatenated_scores = torch.softmax(concatenated_scores, dim=-1)
        final_attn = self.transform(concatenated_scores)
        
        final_attn = self.dropout(final_attn)

        output = torch.einsum('bhlm,bhmd->bhld', final_attn, values.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)

        return (output, None)


class MixCrossAttention(nn.Module):
    def __init__(self, in_channels, out_channels, seq_len_q, seq_len_kv, factor=5, modes=0, mode_select_method='random'):
        super(MixCrossAttention, self).__init__()

        self.factor = factor

        self.transform = nn.Linear(2 * seq_len_q, seq_len_q)
        self.scale = (1 / np.sqrt(out_channels))

        self.dropout = nn.Dropout(0.1)
        self.weights = nn.Parameter(torch.rand(8, seq_len_q, 2))


    
    def forward(self, tf_queries, queries, keys, values, mask):
        # size = [B, L, H, E]
        B, L_Q, H, E = queries.shape
        # _, L_TFQ, _, E = tf_queries.shape
        # _, L_K, _, _ = keys.shape
        _, S, _, D = keys.shape

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


        concatenated_scores = torch.cat([attn_t, attn_tf], dim=-1)
        concatenated_scores = torch.softmax(concatenated_scores, dim=-1)
        final_attn = self.transform(concatenated_scores)

        final_attn = self.dropout(final_attn)

        output = torch.einsum('bhlm,bhmd->bhld', final_attn, values.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)

        return (output, None)