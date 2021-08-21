import torch
from torch import nn, einsum
from math import sqrt
from einops.layers.torch import Rearrange
from copy import copy

from utils.Pos_Encode import get_2dPE_matrix

class MuiltHead_SelfAttention(nn.Module):
    def __init__(self, heads_num, input_dim, output_dim=None, inner_dim=None, qkv_bias=True, dropout=0.0):
        super(MuiltHead_SelfAttention, self).__init__()
        inner_dim = inner_dim if inner_dim is not None else input_dim
        output_dim = output_dim if output_dim is not None else input_dim
        self.inner_dim = inner_dim
        self.heads_num = heads_num
        
        self.W_qkv = nn.Linear(input_dim, inner_dim*heads_num*3, bias=qkv_bias)
        self.div_qkv = Rearrange('b embed_num (qkv heads_num input_dim) -> qkv b heads_num embed_num input_dim', heads_num=heads_num, qkv=3)
        self.softmax = nn.Softmax(dim=-1)
        self.combine = Rearrange('b heads_num embed_num inner_dim -> b embed_num (heads_num inner_dim)', heads_num=heads_num)
        self.out = nn.Sequential(
            nn.Linear(inner_dim*heads_num, output_dim), 
            nn.Dropout(dropout)
        )

    def forward(self, x):
        (q, k, v) = self.div_qkv(self.W_qkv(x))

        a = einsum('bnid, bnjd -> bnij', k, q) / sqrt(self.inner_dim)
        a = self.softmax(a)

        y = einsum('bnij,bnjk -> bnik', a, v)
        y = self.out(self.combine(y))

        return y

class MuiltHead_SelfAttention_Test(nn.Module):
    def __init__(self, heads_num, input_dim, output_dim=None, inner_dim=None, qkv_bias=True):
        super(MuiltHead_SelfAttention_Test, self).__init__()
        inner_dim = inner_dim if inner_dim is not None else input_dim
        output_dim = output_dim if output_dim is not None else input_dim
        self.inner_dim = inner_dim
        self.heads_num = heads_num
        
        self.Wq = nn.Linear(input_dim, inner_dim*heads_num, bias=qkv_bias)
        self.Wk = nn.Linear(input_dim, inner_dim*heads_num, bias=qkv_bias)
        self.Wv = nn.Linear(input_dim, output_dim*heads_num, bias=qkv_bias)
        self.div_kq = Rearrange('b embed_num (heads_num input_dim) -> b heads_num embed_num input_dim', heads_num=heads_num)
        self.div_v = Rearrange('b embed_num (heads_num output_dim) -> b heads_num embed_num output_dim', heads_num=heads_num)
        self.combine_y = Rearrange('b heads_num embed_num output_dim -> b embed_num (heads_num output_dim)', heads_num=heads_num)
        self.softmax = nn.Softmax(dim=1)
        self.concat = nn.Linear(output_dim*heads_num, output_dim)

    
    def forward(self, x):
        (b, _, __) = x.shape
        k = self.div_kq(self.Wk(x))
        q = self.div_kq(self.Wq(x))
        v = self.div_v(self.Wv(x))

        q = q.transpose(2, 3)
        a = self.softmax((k @ q) / sqrt(self.inner_dim))

        y = einsum('bnij,bnjk -> bnik', a, v)

        y = self.concat(self.combine_y(y))
        return y

class Transformer_Encoder(nn.Module):
    def __init__(self, heads_num, embed_num, input_dim, output_dim=None, msa_inner_dim=None, mlp_inner_dim=None, qkv_bias=True, dropout=0.0):
        super(Transformer_Encoder, self).__init__()

        output_dim = output_dim if output_dim is not None else input_dim
        mlp_inner_dim = mlp_inner_dim if mlp_inner_dim is not None else input_dim
        msa_inner_dim = msa_inner_dim if msa_inner_dim is not None else input_dim
        

        self.msa = MuiltHead_SelfAttention(
            heads_num = heads_num,
            input_dim = input_dim,
            output_dim = output_dim,
            inner_dim = msa_inner_dim,
            qkv_bias = qkv_bias,
            dropout = dropout
        )
        self.mlp = nn.Sequential(
            nn.Linear(output_dim, mlp_inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_inner_dim, output_dim),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm((embed_num, output_dim))
        self.norm2 = nn.LayerNorm((embed_num, output_dim))

    def forward(self, x):
        x = self.norm1(self.msa(x)) + x
        x = self.norm2(self.mlp(x)) + x
        return x

class ViT(nn.Module):
    def __init__(self, img_size, patch_size, embed_dim, classes, heads_num, msa_num, msa_inner_dim, mlp_inner_dim, dropout=0.0, embed_dropout=0.0, img_channel=3, dev=None, pos_learnable=False, qkv_bias=True):
        super(ViT, self).__init__()

        self.img_size = img_size
        self.embed_dim = embed_dim
        self.img_channel = img_channel
        self.msa_num = msa_num

        (i_h, i_w) = img_size
        (p_h, p_w) = patch_size
        assert i_h % p_h == 0 and i_w % p_w == 0
        (p_h_num, p_w_num) = (i_h // p_h, i_w // p_w)
        p_num = p_h_num * p_w_num

        self.pos =  nn.Parameter(torch.randn(1, p_h_num*p_w_num, embed_dim)) if pos_learnable else get_2dPE_matrix(p_h_num, p_w_num, embed_dim, dev)
        self.embed_enc = nn.Sequential(
            Rearrange('b c (p_h_num p_h) (p_w_num p_w) -> b (p_h_num p_w_num) (p_h p_w c)', p_h = p_h, p_w = p_w),
            nn.Linear(p_h*p_w*img_channel, embed_dim)
        )
        self.dropout = nn.Dropout(embed_dropout)
        self.transformer = nn.ModuleList([
            Transformer_Encoder(
                heads_num = heads_num,
                embed_num = p_num,
                input_dim = embed_dim,
                output_dim = embed_dim,
                msa_inner_dim = msa_inner_dim,
                mlp_inner_dim = mlp_inner_dim,
                qkv_bias = qkv_bias,
                dropout = dropout
            ) for _ in range(msa_num)
        ])
        self.dec_transform = nn.Linear(p_num, 1)
        self.dec = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, classes)
        )

    def forward(self, x):
        x = self.embed_enc(x) + self.pos
        x = self.dropout(x)
        (bs, embed_num, embed_dim) = x.shape
        for sa in self.transformer:
            x = sa(x)
        x = (self.dec_transform(x.transpose(1, 2))).view(bs, embed_dim)
        x = self.dec(x)
        return x



