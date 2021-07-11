import torch
from torch import nn
from torch.nn.modules.batchnorm import BatchNorm1d
from torch.nn.modules.conv import Conv2d
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_normal_, xavier_uniform_
from math import sqrt

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

class ViT(nn.Module):
    def __init__(self, img_size, embed_dim, classes, img_channel=3, dev=None, sa_num=8, msa_num = 16):
        super(ViT, self).__init__()

        self.img_size = img_size
        self.embed_dim = embed_dim
        self.img_channel = img_channel
        self.msa_num = msa_num

        (h, w) = img_size
        assert h%2 == 0 and w%2 == 0

        self.pos = get_2dPE_matrix(h // 2, w // 2, embed_dim, dev)
        self.embed_enc = nn.Sequential(
            nn.Conv2d(in_channels=img_channel, out_channels=embed_dim, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
        )
        self.transformer = nn.ModuleList(
            [MuiltHead_SelfAttention(sa_num, h*w//4, embed_dim, embed_dim) for _ in range(msa_num)]
        )
        self.BN = nn.ModuleList(
            [nn.BatchNorm1d(self.embed_dim) for _ in range(msa_num)]
        )
        self.relu = nn.ReLU(inplace=True)
        self.dec_matrix = Parameter(torch.empty(1, embed_dim))
        xavier_uniform_(self.dec_matrix)
        self.dec = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(h * w // 4, classes),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        x = self.embed_enc(x)
        x = x + self.pos.permute(2, 0, 1)
        (bs, c, h, w) = x.shape
        x = x.view(bs, c, -1)
        for i in range(self.msa_num):
            t = self.transformer[i](x)
            x = x + t
            #x = self.BN[i](x.transpose(1, 2)).transpose(1, 2)
            x = self.BN[i](x)
            x = self.relu(x)
        x = (self.dec_matrix @ x).view(bs, -1)
        x = self.dec(x)
        return x



