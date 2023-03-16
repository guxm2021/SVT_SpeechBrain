import torch
import torch.nn as nn
from typing import Optional
import numpy as np
import speechbrain as sb
from speechbrain.lobes.models.transformer.Transformer import PositionalEncoding


class RCANet(nn.Module):
    def __init__(
            self,
            nhead,
            d_ffn,
            d_model=None,
            kdim=None,
            vdim=None,
            dropout=0.0,
            activation=nn.ReLU,
            normalize_before=False,
            max_length: Optional[int] = 2500,
            causal=False,
            attention_type="regularMHA",
    ):
        super().__init__()

        self.positional_encoding = PositionalEncoding(d_model, max_length)
        self.layer1 = RCALayer(
                    d_ffn=d_ffn,
                    nhead=nhead,
                    d_model=d_model,
                    kdim=kdim,
                    vdim=vdim,
                    dropout=dropout,
                    activation=activation,
                    normalize_before=normalize_before,
                    causal=causal,
                    attention_type=attention_type)
        
        self.layer2 = RCALayer(
                    d_ffn=d_ffn,
                    nhead=nhead,
                    d_model=d_model,
                    kdim=kdim,
                    vdim=vdim,
                    dropout=dropout,
                    activation=activation,
                    normalize_before=normalize_before,
                    causal=causal,
                    attention_type=attention_type)

    def forward(self, src1, src2,
            src_mask: Optional[torch.Tensor] = None,
            src_key_padding_mask: Optional[torch.Tensor] = None,
            pos_embs: Optional[torch.Tensor] = None,
    ):
        src1 = src1 + self.positional_encoding(src1)
        src2 = src2 + self.positional_encoding(src2)
        
        # attention for modality 1
        output1 = self.layer1(
                src_kv=src1,
                src_q=src2,
                src_mask=src_mask,
                src_key_padding_mask=src_key_padding_mask,
                pos_embs=pos_embs,
        )

        # attention for modality 2
        output2 = self.layer2(
                src_kv=src2,
                src_q=src1,
                src_mask=src_mask,
                src_key_padding_mask=src_key_padding_mask,
                pos_embs=pos_embs,
        )
        return output1, output2


class RCALayer(nn.Module):
    def __init__(
            self,
            d_ffn,
            nhead,
            d_model,
            kdim=None,
            vdim=None,
            dropout=0.0,
            activation=nn.ReLU,
            normalize_before=False,
            attention_type="regularMHA",
            causal=False,
    ):
        super().__init__()

        if attention_type == "regularMHA":
            self.self_att = sb.nnet.attention.MultiheadAttention(
                nhead=nhead,
                d_model=d_model,
                dropout=dropout,
                kdim=kdim,
                vdim=vdim,
            )
            self.cross_att = sb.nnet.attention.MultiheadAttention(
                nhead=nhead,
                d_model=d_model,
                dropout=dropout,
                kdim=kdim,
                vdim=vdim,
            )

        elif attention_type == "RelPosMHAXL":
            self.self_att = sb.nnet.attention.RelPosMHAXL(
                d_model, nhead, dropout, mask_pos_future=causal
            )
            self.cross_att = sb.nnet.attention.RelPosMHAXL(
                d_model, nhead, dropout, mask_pos_future=causal
            )

        self.pos_ffn = sb.nnet.attention.PositionalwiseFeedForward(
            d_ffn=d_ffn,
            input_size=d_model,
            dropout=dropout,
            activation=activation,
        )

        self.norm1 = sb.nnet.normalization.LayerNorm(d_model, eps=1e-6)
        self.norm2 = sb.nnet.normalization.LayerNorm(d_model, eps=1e-6)
        self.dropout_self_attn = torch.nn.Dropout(dropout)
        self.dropout_cross_attn = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)
        self.normalize_before = normalize_before

    def forward(self, src_kv, src_q,
            src_mask: Optional[torch.Tensor] = None,
            src_key_padding_mask: Optional[torch.Tensor] = None,
            pos_embs: Optional[torch.Tensor] = None,
    ):
        if self.normalize_before:
            src = self.norm1(src_kv)
        else:
            src = src_kv

        self_attn_output, _ = self.self_att(
            query=src,
            key=src,
            value=src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            pos_embs=pos_embs,
        )

        cross_attn_output, _ = self.cross_att(
            query=src_q,
            key=src,
            value=src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            pos_embs=pos_embs,
        )

        # add & norm
        src = src_kv + self.dropout_self_attn(self_attn_output) + self.dropout_cross_attn(cross_attn_output)
        if not self.normalize_before:
            src = self.norm1(src)

        if self.normalize_before:
            src = self.norm2(src)
        else:
            src = src
        output = self.pos_ffn(src)

        # add & norm
        output = src + self.dropout2(output)
        if not self.normalize_before:
            output = self.norm2(output)

        return output


class FusionRCA(nn.Module):
    def __init__(self, nhead=8, d_ffn=3072, d_model=1024):
        super().__init__()
        self.fusion = RCANet(nhead=nhead, 
                             d_ffn=d_ffn, d_model=d_model)

    
    def forward(self, audio_feats, video_feats):
        # audio_feats: (B, T1, 1024)  video_feats: (B, T2, 1024)
       
        # frame alignment
        audio_batch, audio_frame, audio_feat_dim = audio_feats.size()
        video_batch, video_frame, video_feat_dim = video_feats.size()
        diff = audio_frame - video_frame
        if diff < 0:
            video_feats = video_feats[:, :diff]
        elif diff > 0:
            pad = torch.zeros([video_batch, diff, video_feat_dim]).type_as(video_feats).to(audio_feats.device)
            video_feats = torch.cat([video_feats, pad], dim=1)  # time axis
        if abs(diff) > 15:
            print("Alignment is wrong")
        
        # feature fusion
        audio_feats, video_feats = self.fusion(audio_feats, video_feats)
        feats = audio_feats + video_feats
        return feats