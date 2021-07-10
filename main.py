from model.SelfAttention import SelfAttention
import torch
from torch import nn

Net = SelfAttention(embed_num=24, input_dim=16, inner_dim=2, output_dim=4)
l = nn.L1Loss()
opt = torch.optim.Adam(Net.parameters(), lr=3e-4)