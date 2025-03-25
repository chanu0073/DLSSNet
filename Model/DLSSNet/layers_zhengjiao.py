import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
import numpy as np

from torch import Tensor
from einops import rearrange, reduce, repeat

import torch.nn.functional as F


import torch.nn.functional as F
from torch.ao.nn.quantizable import MultiheadAttention

class Conv2dWithConstraint(nn.Conv2d):
    '''
    Lawhern V J, Solon A J, Waytowich N R, et al. EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces[J]. Journal of neural engineering, 2018, 15(5): 056013.
    '''
    def __init__(self, *args, p = 2, doWeightNorm = True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        self.p = p
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm: 
            self.weight.data = torch.renorm(
                self.weight.data, p=self.p, dim=0, maxnorm=self.max_norm
            )
        return super(Conv2dWithConstraint, self).forward(x)

class PatchEmbedding(nn.Module):
    def __init__(self, len_window, num_channel, d_model, frame_stride, dropout) -> None:
        super().__init__()
        self.dropout = 0.25

        self.norm = nn.BatchNorm2d(d_model)
        self.TaskPotentialConvBlock = nn.Sequential(
            nn.ZeroPad2d((len_window-1, 0, 0, 0)),
            Conv2dWithConstraint(in_channels=1, 
                      out_channels=d_model, 
                      kernel_size=(num_channel, 1), 
                      bias=True,
                      p = 1,
                      doWeightNorm=True,
                      max_norm=1), 
            # nn.BatchNorm2d(d_model),
            nn.ELU()
            )
        self.TemporalStateConvBlock = nn.Sequential(
            Conv2dWithConstraint(in_channels=d_model, 
                      out_channels=d_model, 
                      kernel_size=(1, len_window), 
                      stride=frame_stride, 
                      bias=True,
                      p=2,
                      max_norm=0.25), 
            nn.BatchNorm2d(d_model), 
            nn.ELU(), 
            nn.Dropout(self.dropout)
            )
        self.project = nn.Conv2d(in_channels=d_model, 
                                 out_channels=d_model, 
                                 kernel_size=(1,1),
                                 stride=(1,1),
                                 bias=True)
    def forward(self, x):
        # x1 = self.D(torch.transpose(x, -1, -2))
        x = torch.unsqueeze(x, 1)
        
                # x:[batch, channelnum, timepoints]->[batch, 1, channelnum, timepoints]
        x = self.TaskPotentialConvBlock(x)
                # x:[batch, num_kernel, 1, timepoints]
        x = self.TemporalStateConvBlock(x)
                # x:[batch, num_kernel, 1, num_frame]
        x = self.project(x)
        
        x = torch.squeeze(x)
                # x:[batch, num_kernel, num_frame]
        x = torch.transpose(x, 1, 2)
                # x:[batch, num_frame, num_kernel]
        return x

# class PatchEmbedding(nn.Module):
#     def __init__(self, len_window, num_channel, d_model, frame_stride, dropout):
#         # self.patch_size = patch_size
#         super().__init__()

#         self.shallownet = nn.Sequential(
#             nn.Conv2d(1, 40, (1, len_window), (1, frame_stride)),
#             nn.Conv2d(40, 40, (num_channel, 1), (1, 1)),
#             nn.BatchNorm2d(40),
#             nn.ELU(),
#             # nn.AvgPool2d((1, 75), (1, 15)),  # pooling acts as slicing to obtain 'patch' along the time dimension as in ViT
#             nn.Dropout(0.5),
#         )

#         self.projection = nn.Sequential(
#             nn.Conv2d(40, d_model, (1, 1), stride=(1, 1)),  # transpose, conv could enhance fiting ability slightly
#             # Rearrange('b e (h) (w) -> b (h w) e'),
#         )


#     def forward(self, x: Tensor) -> Tensor:
#         x = torch.unsqueeze(x, 1)
#         # b, _, _, _ = x.shape
#         x = self.shallownet(x)
#         x = self.projection(x)
#         x = torch.squeeze(x)
#                 # x:[batch, num_kernel, num_frame]
#         x = torch.transpose(x, 1, 2)
#                 # x:[batch, num_frame, num_kernel]
#         return x

class PoswiseFeedForwardNet(Module):
# 前馈神经网络。
# 输入inputs ，经过两个全连接成，得到的结果再加上 inputs ，再做LayerNorm归一化。LayerNorm归一化可以理解层是把Batch中每一句话进行归一化。
    def __init__(self, d_model:int = 64, d_ff:int = 256):
        super(PoswiseFeedForwardNet, self).__init__()
        self.d_model = d_model
        self.layer_norm = nn.LayerNorm(d_model)
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.GELU(),
            nn.Linear(d_ff, d_model, bias=False))
        
    def forward(self, inputs):                             # inputs: [batch_size, seq_len, d_model]
        residual = inputs
        output = self.fc(inputs)
        return self.layer_norm(output + residual)   # [batch_size, seq_len, d_model]  

# -------------------------------------------------------------
# Graph U Net 中的gPooling
class Pool(nn.Module):

    def __init__(self, dropout_p, in_dim, low_p):
        super(Pool, self).__init__()
        self.low_p = low_p
        self.softmax = nn.Softmax(dim = -1)
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(in_dim, 2)
        self.drop = nn.Dropout(p=dropout_p) if dropout_p > 0 else nn.Identity()

    def forward(self, h):
        Z = self.drop(h)
        a = self.proj.get_parameter('weight')
        weights = self.proj(Z).squeeze()
        scores = self.softmax(weights)
        scaler = self.sigmoid(weights[:,:,0])
        mask, mask_index = p_filter(scores=scores, low_p=self.low_p)
        Z = torch.mul(Z, torch.unsqueeze(scaler, -1))
        return Z, mask, mask_index,scaler, scores
    
# class Unpool(nn.Module):

#     def __init__(self, *args):
#         super(Unpool, self).__init__()

#     def forward(self, g, h, pre_h, idx):
#         new_h = h.new_zeros([g.shape[0], h.shape[1]])
#         new_h[idx] = h
#         return g, new_h

def p_filter(scores, low_p):
    mask_indx = torch.gt(scores, low_p)
    mask_indx.requires_grad = False
    
    mask_indx = torch.unsqueeze(mask_indx[:,:,1], -1)
    mask_indxforD = ~mask_indx
    mask_indxforD = mask_indxforD.float()
    mask_indx = mask_indx.float()
    mask = torch.bmm(mask_indx, torch.transpose(mask_indx, -1, -2))
    mask = mask.bool()
    return mask, mask_indxforD

    

# def top_k_graph(scores, g, h, k):
#     batchsize = h.shape[0]
#     num_nodes = g.shape[1]
#     new_h_batch = None
    
#     for i in range(0, batchsize):
#         values, idx = torch.topk(scores[i].squeeze(), max(2, k))
#         new_h = h[i,idx, :]
#         values = torch.unsqueeze(values, -1)
#         new_h = torch.mul(new_h, values)
#         new_h = new_h.unsqueeze(0)
#         if new_h_batch is None:
#             new_h_batch = new_h
#         else:
#             new_h_batch = torch.concatenate((new_h_batch, new_h), dim=0)
    
    
    # un_g = g.bool().float()
    # un_g = torch.matmul(un_g, un_g).bool().float()
    # un_g = un_g[idx, :]
    # un_g = un_g[:, idx]
    # g = norm_g(un_g)
    # return new_h_batch

# def norm_g(g):
#     degrees = torch.sum(g, 1)
#     g = g / degrees
#     return g

# 这部分是对参数初始化
class Initializer(object):

    @classmethod
    def _glorot_uniform(cls, w):
        if len(w.size()) == 2:
            fan_in, fan_out = w.size()
        elif len(w.size()) == 3:
            fan_in = w.size()[1] * w.size()[2]
            fan_out = w.size()[0] * w.size()[2]
        else:
            fan_in = np.prod(w.size())
            fan_out = np.prod(w.size())
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        w.uniform_(-limit, limit)

    @classmethod
    def _param_init(cls, m):
        if isinstance(m, nn.parameter.Parameter):
            cls._glorot_uniform(m.data)
        elif isinstance(m, nn.Linear):
            m.bias.data.zero_()
            cls._glorot_uniform(m.weight.data)

    @classmethod
    def weights_init(cls, m):
        for p in m.modules():
            if isinstance(p, nn.ParameterList):
                for pp in p:
                    cls._param_init(pp)
            else:
                cls._param_init(p)

        for name, p in m.named_parameters():
            if '.' not in name:
                cls._param_init(p)
# ---------------------------------------------------------------------------

class EncoderLayer(nn.Module):
    def __init__(self, 
                 d_model:int = 64, 
                 d_k:int = 32, 
                 n_heads:int = 2, 
                 d_ff:int = 256, 
                 dropout = 0.1,
                 low_p: float = 0.5) -> None:
        super().__init__()
        self.nhead = n_heads
        self.enc_self_attn = MultiHeadAttention_Enc(emb_size=d_model,
                                                    num_heads= n_heads,
                                                    dropout=dropout)
        # self.enc_self_attn = MultiheadAttention(
        #                             embed_dim = d_model, 
        #                             num_heads = n_heads,
        #                             dropout=dropout,
        #                             bias = True,
        #                             kdim=d_k,
        #                             vdim=d_k,
        #                             batch_first=True
        #                             )
        self.layernorm = nn.LayerNorm(d_model)
        self.pool = Pool(dropout_p=dropout, in_dim=d_model, low_p=low_p)
        Initializer.weights_init(self.pool)
        self.pos_ffn = PoswiseFeedForwardNet(d_model=d_model, d_ff=d_ff)

    def forward(self, x):
        x, mask, mask_index,scaler, scores = self.pool(x)
        pooled_x = x.clone()
        # x = self.layernorm(x)
        mask = torch.repeat_interleave(mask, self.nhead, dim = 0)
        # 
        x, attn = self.enc_self_attn(x, x, x, mask = mask)
        x = self.pos_ffn(x)
        return x, attn,mask_index, pooled_x,scaler,scores
    

class DecoderLayer(nn.Module):
    def __init__(self,
                 d_model:int = 64, 
                 d_k: int=64,
                 n_heads:int = 1,
                 d_ff: int = 2,
                 dropout = 0.1):
        super().__init__()
        self.dec_self_attn = MultiHeadAttention_Dec(
                                    emb_size = d_model, 
                                    num_heads = n_heads,
                                    dropout=dropout
                                    )
        self.pos_ffn = PoswiseFeedForwardNet(d_model=d_model, d_ff=d_ff)
    def forward(self, vectors, states):
        y, attn = self.dec_self_attn(vectors, states, states)
        y = self.pos_ffn(y)
        return y, attn

# class ClassificationHead(nn.Module):
#     def __init__(self, num_frame, emb_size, n_classes):
#         super().__init__()
#         self.conv = nn.Conv1d(in_channels=num_frame, out_channels=1, kernel_size=1, stride=1)
#         self.elu = nn.ELU()
#         self.linear = nn.Linear(in_features=emb_size, out_features=n_classes)
#     def forward(self,x):
#         x = self.elu(self.conv(x))

#         x = torch.squeeze(x)
#         x = self.linear(x)
#         return x

    
class ClassificationHead(nn.Sequential):
    def __init__(self, num_frame, emb_size, n_classes):
        super().__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(num_frame*emb_size, 256),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(32, n_classes)
        )

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        out = self.fc(x)
        return out
    
class ClassificationTransHead(nn.Module):
    def __init__(self, d_model, d_k, num_heads, dropout,  state_classes, n_classes) -> None:
        super().__init__()
        self.summary_token = nn.Parameter(torch.Tensor(1, state_classes, d_model))
        # self.classATTN = MultiHeadAttention_Enc(emb_size=d_model,
        #                                             num_heads= num_heads,
        #                                             dropout=dropout)
        self.classATTN = MultiheadAttention(
                                    embed_dim = d_model, 
                                    num_heads = num_heads,
                                    dropout=dropout,
                                    bias = True,
                                    kdim=d_k,
                                    vdim=d_k,
                                    batch_first=True
                                    )
        # self.layernorm = nn.LayerNorm(d_model)
        self.fc = nn.Sequential(
            nn.Linear(state_classes*d_model, 256),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(32, n_classes)
        )
        # self.fc = nn.Linear(state_classes*d_model, n_classes)
        torch.nn.init.orthogonal_(self.summary_token)
    
    def forward(self, x):
        summary_token = self.summary_token.repeat((x.shape[0], 1, 1))
        # x = self.layernorm(x)
        x = torch.cat([x,summary_token], dim=1) 
        x, attn = self.classATTN(x, x, x)
        indx = -summary_token.shape[1]
        states = x[:, indx:, :].contiguous()
        out = states.view(x.size(0), -1)
        out = self.fc(out)
        # states = states.contiguous()
        # out = states.view(states.size(0), -1)
        # out = self.fc(out)
        return out, summary_token, attn, states


class MultiHeadAttention_Dec(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, q: Tensor, k, v, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(q), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(k), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(v), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.masked_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = energy / scaling
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out, att

class MultiHeadAttention_Enc(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, q: Tensor, k, v, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(q), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(k), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(v), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            mask = torch.unsqueeze(mask, dim= 1)
            energy = energy.masked_fill(mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out, att

