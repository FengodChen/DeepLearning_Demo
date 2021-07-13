import torch
from torch import nn
from math import sqrt
from einops.layers.torch import Rearrange

from model.Pos_Encode import get_2dPE_matrix

class MuiltHead_SelfAttention(nn.Module):
    '''
    See `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`

    Example::
        >>> x = torch.rand(batch_size, input_dim, embed_num)
        >>> Net = MuiltHead_SelfAttention(sa_num, embed_num, input_dim, output_dim)
        >>> y = Net(x)

    Man::
        x.shape = (batch_size, embed_num, input_dim)
        Wq.shape = (input_dim, inner_dim)
        Wk.shape = (input_dim, inner_dim)
        Wv.shape = (embed_num, outut_dim)
        y.shape = (batch_size, embed_num, output_dim)

    Algorithm::
        Input: x

        k = (x @ Wk) + bias_k 
        q = (x @ Wq) + bias_q
        a = softmax((k @ q.tranpose) / sqrt(inner_dim))
        y = (x @ Wv) + bias_v

        Output: y
    '''
    def __init__(self, sa_num, embed_num, input_dim, output_dim, inner_dim=None, qkv_bias=True):
        super(MuiltHead_SelfAttention, self).__init__()
        inner_dim = inner_dim if inner_dim is not None else input_dim
        self.inner_dim = inner_dim
        self.sa_num = sa_num
        
        self.Wq = nn.Linear(input_dim, inner_dim*sa_num, bias=qkv_bias)
        self.Wk = nn.Linear(input_dim, inner_dim*sa_num, bias=qkv_bias)
        self.Wv = nn.Linear(embed_num*sa_num, output_dim*sa_num, bias=qkv_bias)
        self.div_kq = Rearrange('b embed_num (sa_num input_dim) -> (b sa_num) embed_num input_dim', sa_num=sa_num)
        self.combine_a = Rearrange('(b sa_num) embed_num_h embed_num_w -> b embed_num_h (sa_num embed_num_w)', sa_num=sa_num)
        self.softmax = nn.Softmax(dim=1)
        self.concat = nn.Linear(output_dim*sa_num, output_dim)

    
    def forward(self, x):
        (b, _, __) = x.shape
        k = self.div_kq(self.Wk(x))
        q = self.div_kq(self.Wq(x))
        q = q.transpose(1, 2)
        #a = (k @ q) / sqrt(self.inner_dim)
        a = self.softmax((k @ q) / sqrt(self.inner_dim))
        v = self.Wv(self.combine_a(a))
        y = self.concat(v)
        return y

class Transformer_Encoder(nn.Module):
    def __init__(self, sa_num, embed_num, input_dim, output_dim, inner_dim=None, qkv_bias=True):
        super(Transformer_Encoder, self).__init__()

        self.msa = MuiltHead_SelfAttention(sa_num, embed_num, input_dim, output_dim, inner_dim, qkv_bias)
        self.net = nn.Sequential(
            nn.Linear(output_dim, 16),
            nn.GELU(),
            nn.Linear(16, output_dim)
        )
        self.norm1 = nn.LayerNorm((embed_num, output_dim))

        self.w1 = nn.Linear(output_dim, output_dim)
        self.relu = nn.ReLU(inplace=True)
        self.w2 = nn.Linear(output_dim, output_dim)

        self.norm2 = nn.LayerNorm((embed_num, output_dim))

    def forward(self, x):
        x = self.norm1(self.msa(x)) + x
        x = self.norm2(self.net(x)) + x
        '''
        t = self.msa(x)
        x = self.norm1(x + t)

        t = self.relu(self.w1(x))
        t = self.w2(t)

        x = self.norm2(x + t)
        '''
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
        p_num = p_h_num * p_w_num

        #self.pos = get_2dPE_matrix(p_h_num, p_w_num, embed_dim, dev)
        self.pos = nn.Parameter(torch.randn(1, p_h_num*p_w_num, embed_dim))
        self.embed_enc = Rearrange('b c (p_h_num p1) (p_w_num p2) -> b (p_h_num p_w_num) (p1 p2 c)', p1 = p_h, p2 = p_w)
        self.enc_transform = nn.Linear(p_h*p_w*img_channel, embed_dim)
        self.transformer = nn.ModuleList([
            Transformer_Encoder(sa_num, p_num, embed_dim, embed_dim) for _ in range(msa_num)
        ])
        self.dec_transform = nn.Linear(p_num, 1)
        self.dec = nn.Sequential(
            #nn.ReLU(inplace=True), 
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, classes),
            #nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.embed_enc(x)
        x = self.enc_transform(x)
        x = x + self.pos
        (bs, embed_dim, n) = x.shape
        for sa in self.transformer:
            x = sa(x)
        x = (self.dec_transform(x.transpose(1, 2))).view(bs, self.embed_dim)
        x = self.dec(x)
        return x



