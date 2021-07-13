import torch
from torch import nn
from torch.nn.modules.batchnorm import BatchNorm1d
from torch.nn.modules.conv import Conv2d
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_normal_, xavier_uniform_
from math import sqrt
from einops.layers.torch import Rearrange

from model.Pos_Encode import get_2dPE_matrix

class SelfAttention(nn.Module):
    '''
    See `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`

    Example::
        >>> x = torch.rand(batch_size, input_dim, embed_num)
        >>> Net = SelfAttention(embed_num, input_dim, output_dim)
        >>> y = Net(x)

    Man::
        x.shape = (batch_size, input_dim, embed_num)
        Wq.shape = (inner_dim, input_dim)
        Wk.shape = (inner_dim, input_dim)
        Wv.shape = (outut_dim, embed_num)
        y.shape = (batch_size, output_dim, embed_num)

    Algorithm::
        Input: x

        k = (Wk @ x) + bias_k 
        q = (Wq @ x) + bias_q
        a = softmax((k.tranpose @ q) / sqrt(inner_dim))
        y = (Wv @ a) + bias_v

        Output: y
    '''
    def __init__(self, embed_num, input_dim, output_dim, inner_dim=None, qkv_bias=True):
        super(SelfAttention, self).__init__()
        inner_dim = inner_dim if inner_dim is not None else input_dim
        self.inner_dim = inner_dim
        self.qkv_bias = qkv_bias

        self.Wq = Parameter(torch.empty((inner_dim, input_dim)))
        self.Wk = Parameter(torch.empty((inner_dim, input_dim)))
        self.Wv = Parameter(torch.empty((output_dim, embed_num)))
        self.softmax = nn.Softmax(dim=1)
        if (qkv_bias):
            self.q_bias = Parameter(torch.empty((1, 1)))
            self.k_bias = Parameter(torch.empty((1, 1)))
            self.v_bias = Parameter(torch.empty((1, 1)))

        self._reset_parameters()
    
    def _reset_parameters(self):
        xavier_uniform_(self.Wq)
        xavier_uniform_(self.Wk)
        xavier_uniform_(self.Wv)
        if (self.qkv_bias):
            xavier_normal_(self.q_bias)
            xavier_normal_(self.k_bias)
            xavier_normal_(self.v_bias)
    
    def forward(self, x):
        k = self.Wk @ x
        q = self.Wq @ x
        if (self.qkv_bias):
            k = k + self.k_bias
            q = q + self.q_bias
        k = k.transpose(1, 2)
        a = self.softmax((k @ q) / sqrt(self.inner_dim))
        v = self.Wv @ a
        if (self.qkv_bias):
            v = v + self.v_bias
        return v

class MuiltHead_SelfAttention(nn.Module):
    '''    
    Example::
        >>> x = torch.rand(batch_size, input_dim, embed_num)
        >>> Net = MuiltHead_SelfAttention(sa_num, embed_num, input_dim, output_dim)
        >>> y = Net(x)

    Man::
        sa_num: self-attention number
        others args: read man of SelfAttention class
    '''
    def __init__(self, sa_num, embed_num, input_dim, output_dim, inner_dim=None, qkv_bias=True):
        super(MuiltHead_SelfAttention, self).__init__()
        self.msa_list = nn.ModuleList([SelfAttention(embed_num, input_dim, output_dim, inner_dim, qkv_bias) for _ in range(sa_num)])
        self.concat_matrix = Parameter(torch.empty(output_dim, output_dim*sa_num))
        xavier_uniform_(self.concat_matrix)
        self.sa_num = sa_num
    
    def forward(self, x):
        y = None
        for sa in self.msa_list:
            y_i = sa(x)
            y = torch.cat((y, y_i), dim=1) if y is not None else y_i
        y = self.concat_matrix @ y
        return y

class Transformer_Encoder(nn.Module):
    def __init__(self, sa_num, embed_num, input_dim, output_dim, inner_dim=None, qkv_bias=True):
        super(Transformer_Encoder, self).__init__()

        self.msa = MuiltHead_SelfAttention(sa_num, embed_num, input_dim, output_dim, inner_dim, qkv_bias)
        self.norm1 = nn.LayerNorm((output_dim, embed_num))

        self.w1 = Parameter(torch.empty((output_dim, output_dim)))
        self.b1 = Parameter(torch.empty((1, 1)))
        self.relu = nn.ReLU(inplace=True)
        self.w2 = Parameter(torch.empty((output_dim, output_dim)))
        self.b2 = Parameter(torch.empty((1, 1)))

        self.norm2 = nn.LayerNorm((output_dim, embed_num))

        xavier_uniform_(self.w1)
        xavier_uniform_(self.w2)
        xavier_normal_(self.b1)
        xavier_normal_(self.b2)
    
    def forward(self, x):
        t = self.msa(x)
        x = self.norm1(x + t)

        t = self.relu((self.w1 @ x) + self.b1)
        t = (self.w2 @ t) + self.b2

        x = self.norm2(x + t)
        return x

class ViT(nn.Module):
    def __init__(self, img_size, patch_size, embed_dim, classes, img_channel=3, dev=None, sa_num=4, msa_num = 8):
        super(ViT, self).__init__()

        self.img_size = img_size
        self.embed_dim = embed_dim
        self.img_channel = img_channel
        self.msa_num = msa_num

        (i_h, i_w) = img_size
        (p_h, p_w) = patch_size
        assert i_h % p_h == 0 and i_w % p_w == 0
        (p_h_num, p_w_num) = (i_h // p_h, i_w // p_w)

        self.pos = get_2dPE_matrix(p_h_num, p_w_num, embed_dim, dev)
        print(self.pos.shape)
        self.embed_enc = Rearrange('b c (p_h_num p1) (p_w_num p2) -> b (p1 p2 c) (p_h_num p_w_num)', p1 = p_h, p2 = p_w)
        self.enc_matrix = Parameter(torch.empty(embed_dim, p_h*p_w*img_channel))
        self.transformer = nn.ModuleList([
            Transformer_Encoder(sa_num, p_h_num*p_w_num, embed_dim, embed_dim) for _ in range(msa_num)
        ])
        self.relu = nn.ReLU(inplace=True)
        self.dec_matrix = Parameter(torch.empty(1, embed_dim))
        self.dec = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(p_h_num * p_w_num, classes),
            nn.Softmax(dim=1)
        )

        xavier_uniform_(self.enc_matrix)
        xavier_uniform_(self.dec_matrix)
    
    def forward(self, x):
        x = self.embed_enc(x)
        x = self.enc_matrix @ x
        x = x + self.pos
        (bs, embed_dim, n) = x.shape
        #x = x.view(bs, c, -1)
        for sa in self.transformer:
            x = sa(x)
        x = (self.dec_matrix @ x).view(bs, -1)
        x = self.dec(x)
        return x



