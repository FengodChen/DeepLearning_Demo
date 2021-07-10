import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_normal_, xavier_uniform_


class SelfAttention(nn.Module):
    def __init__(self, embed_num, input_dim, output_dim, inner_dim=None, qkv_bias=True):
        super(SelfAttention, self).__init__()
        inner_dim = inner_dim if inner_dim is not None else input_dim
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
        a = self.softmax(k @ q)
        v = self.Wv @ a
        if (self.qkv_bias):
            v = v + self.v_bias
        return v


Net = SelfAttention(embed_num=24, input_dim=16, inner_dim=2, output_dim=4)
l = nn.L1Loss()
opt = torch.optim.Adam(Net.parameters(), lr=3e-4)




