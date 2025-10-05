import torch
import torch.nn as nn
from torch import Tensor
# from Model.EEGConvstateFormerwithGPoolingV2.layers import PatchEmbedding, EncoderLayer, DecoderLayer, ClassificationTransHead
from Model.DLSSNet.layers_zhengjiao import PatchEmbedding, EncoderLayer, DecoderLayer, ClassificationTransHead


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()

        # Compute the positional encodings in log space
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        pos_enc = torch.zeros((max_len, d_model))

        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)

        pos_enc = pos_enc.unsqueeze(0)
        self.register_buffer('pos_enc', pos_enc)

    def forward(self, x):
        return x + self.pos_enc[:, :x.size(1)].detach()
    
class Encoder(nn.Module):
    def __init__(self, 
                 num_layers: int = 6, 
                 d_model: int=64, 
                 d_k:int=32, 
                 num_heads:int=2, 
                 dropout=0.1, 
                 d_ffrate:int = 4, 
                 max_len: int= 20,
                 low_p:float = 0.5):
        super(Encoder, self).__init__()
        d_ff = d_ffrate*d_model
        self.posenc = PositionalEncoding(d_model=d_model, max_len=max_len)
        self.layers = nn.ModuleList([EncoderLayer(d_model = d_model, 
                                                  d_k = d_k, 
                                                  n_heads = num_heads, 
                                                  d_ff = d_ff,  
                                                  dropout = dropout,
                                                  low_p = low_p) 
                                                  for _ in range(num_layers)])
        
    def forward(self, x):
        enc_self_attns = []
        mask_indexs = []
        x = self.posenc(x)
        for layer in self.layers:
            x, enc_self_attn, mask_index, pooled_X,scaler,scores = layer(x)  # enc_outputs :   [batch_size, src_len, d_model], 
                                                                                 # enc_self_attn : [batch_size, n_heads, src_len, src_len]
            enc_self_attns.append(enc_self_attn) 
            mask_indexs.append(mask_index)

        return x, enc_self_attns[0], mask_indexs[0],pooled_X,scaler,scores
    
class Net(nn.Module):
    def __init__(self, 
                 num_channels,
                 len_window,
                 d_model,
                 frame_stride, 
                 num_frame, 
                 num_head, 
                 encoder_num_layers, 
                 low_p,
                 dropout, 
                 transformerparwiseforward_dimrat,
                 statenum,
                 num_class) -> None:
        super().__init__()
        self.Embedding = PatchEmbedding(
                            len_window=len_window, 
                            num_channel=num_channels,
                            d_model=d_model,
                            frame_stride=frame_stride,
                            dropout=dropout)
        self.Encoder = Encoder(num_layers = encoder_num_layers,
                               d_model = d_model, 
                               d_k=d_model, 
                               num_heads = num_head, 
                               dropout = dropout, 
                               d_ffrate=transformerparwiseforward_dimrat, 
                               max_len=num_frame,
                               low_p=low_p)
        d_ff = transformerparwiseforward_dimrat*d_model
        self.Decoder = DecoderLayer(d_model=d_model, d_k=d_model, n_heads=num_head, d_ff=d_ff, dropout=dropout)
        self.classhead = ClassificationTransHead(d_model=d_model,
                                                 d_k=d_model, 
                                                 num_heads=num_head,
                                                 dropout=dropout,
                                                 state_classes=statenum,
                                                 n_classes=num_class)
        
    def forward(self, x):
        x = x.to(torch.float32)
        x = self.Embedding(x)
        embeded_x = torch.detach(x)
        x, attn, mask_index, Pooled_x,scaler,scores = self.Encoder(x)
        y = torch.mul(embeded_x, mask_index)
        out, basis, attn_class, Encoded_x = self.classhead(x)
        dec_x, dec_attn = self.Decoder(y, basis)
        # dec_x = torch.mul(dec_x, mask_index)
        all_att = {'encoder_att': attn, 'state_attn':attn_class, 'decoder_attn': dec_attn}
        
        return all_att, out, dec_x, embeded_x, basis, Pooled_x, Encoded_x, scaler, scores
        # return out

        