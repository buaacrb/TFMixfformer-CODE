import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed_tf import DataEmbedding, DataEmbedding_wo_pos
# from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.TFMixCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.TFMixAttention import MixSelfAttention, MixCrossAttention
from layers.TFMixformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp, series_decomp_multi
from utils.timefreqfeatures import timefreqfeatures
import numpy as np


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Model(nn.Module):
    """
    TFMixformer performs the attention mechanism on time and frequency domain
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.version = configs.version
        self.mode_select = configs.mode_select
        self.modes = configs.modes
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.num_freq = configs.num_freq
        self.samp_freq = configs.samp_freq

        # Decomp
        kernel_size = configs.moving_avg
        if isinstance(kernel_size, list):
            self.decomp = series_decomp_multi(kernel_size)
        else:
            self.decomp = series_decomp(kernel_size)

        if configs.features == 'S':
            configs.enc_in = 1
            configs.dec_in = 1

        self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.num_freq
, configs.d_model, configs.seq_len, configs.embed, configs.freq,
                                                  configs.dropout)
        self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.num_freq
, configs.d_model, configs.seq_len//2+configs.pred_len, configs.embed, configs.freq,
                                                  configs.dropout)

        encoder_self_att = MixSelfAttention(in_channels=configs.d_model,
                                        out_channels=configs.d_model,
                                        seq_len=self.seq_len,
                                        factor=configs.factor,
                                        modes=configs.modes,
                                        mode_select_method=configs.mode_select)
        decoder_self_att = MixSelfAttention(in_channels=configs.d_model,
                                        out_channels=configs.d_model,
                                        seq_len=self.seq_len//2+self.pred_len,
                                        factor=configs.factor,
                                        modes=configs.modes,
                                        mode_select_method=configs.mode_select)
        decoder_cross_att = MixCrossAttention(in_channels=configs.d_model,
                                                    out_channels=configs.d_model,
                                                    seq_len_q=self.seq_len//2+self.pred_len,
                                                    seq_len_kv=self.seq_len,
                                                    factor=configs.factor,
                                                    modes=configs.modes,
                                                    mode_select_method=configs.mode_select)
        # Encoder
        enc_modes = int(min(configs.modes, configs.seq_len//2))
        dec_modes = int(min(configs.modes, (configs.seq_len//2+configs.pred_len)//2))
        print('enc_modes: {}, dec_modes: {}'.format(enc_modes, dec_modes))

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        encoder_self_att,
                        configs.d_model, configs.n_heads),

                    configs.d_model,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        decoder_self_att,
                        configs.d_model, configs.n_heads),
                    AutoCorrelationLayer(
                        decoder_cross_att,
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.c_out,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]]).to(device)  # cuda()
        seasonal_init, trend_init = self.decomp(x_enc)
        # decoder input
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        
        _, x_enc_amp, x_enc_pha = timefreqfeatures(x_enc, self.samp_freq, self.num_freq)
        _, x_dec_amp, x_dec_pha = timefreqfeatures(seasonal_init, self.samp_freq, self.num_freq)
        x_enc_amp = x_enc_amp.to(x_enc.device)
        x_enc_pha = x_enc_pha.to(x_enc.device)
        x_dec_amp = x_dec_amp.to(seasonal_init.device)
        x_dec_pha = x_dec_pha.to(seasonal_init.device)

        x_dec_amp = F.pad(x_dec_amp[:, -self.label_len:, :], (0, 0, 0, self.pred_len))
        x_dec_pha = F.pad(x_dec_pha[:, -self.label_len:, :], (0, 0, 0, self.pred_len))

        seasonal_init = F.pad(seasonal_init[:, -self.label_len:, :], (0, 0, 0, self.pred_len))
        # enc
        enc_out, enc_out_tf = self.enc_embedding(x_enc, x_enc_amp, x_enc_pha, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, enc_out_tf, attn_mask=enc_self_mask)
        # dec
        dec_out, dec_out_tf = self.dec_embedding(seasonal_init, x_dec_amp, x_dec_pha, x_mark_dec)
        seasonal_part, trend_part = self.decoder(dec_out, dec_out_tf, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask,
                                                 trend=trend_init)
        # final
        dec_out = trend_part + seasonal_part

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
