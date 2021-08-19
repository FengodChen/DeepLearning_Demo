import torch
from torch import nn, einsum
from math import sqrt
from einops.layers.torch import Rearrange
from copy import copy
from timm.models.layers import to_2tuple

class Embed_Encoder(nn.Module):
	def __init__(self, img_size, patch_size, img_channel, embed_dim):
		super().__init__()

		self.img_size = to_2tuple(img_size)
		self.patch_size = to_2tuple(patch_size)

		img_h, img_w = self.img_size
		patch_h, patch_w = self.patch_size

		assert img_h % patch_h == 0 and img_w % patch_w == 0

		self.img_channel = img_channel
		self.embed_dim = embed_dim

		patch_h_num, patch_w_num = img_h // patch_h, img_w // patch_w
		self.patch_num = patch_h_num * patch_w_num

		self.embed_enc = nn.Sequential(
			nn.Conv2d(img_channel, embed_dim, kernel_size=self.patch_size, stride=self.patch_size),
			Rearrange('b c h w-> b (h w) c'),
			nn.LayerNorm(embed_dim)
		)
		
	def forward(self, x):
		(B, C, H, W) = x.shape
		assert H == self.img_size[0] and W == self.img_size[1]
		return self.embed_enc(x)

class Embed_Pooling(nn.Module):
	def __init__(self, input_size, output_size, input_dim, output_dim):
		super().__init__()
		self.input_size = to_2tuple(input_size)
		self.output_size = to_2tuple(output_size)
		self.input_dim = input_dim
		self.output_dim = output_dim
		
		input_h, input_w = self.input_size
		output_h, output_w = self.output_size

		assert input_h % output_h == 0 and input_w % output_w == 0
		patch_h, patch_w = input_h // output_h, input_w // output_w
		patch_size = (patch_h, patch_w)

		self.pooling = nn.Sequential(
			Rearrange('b (h w) c -> b c h w'),
			nn.Conv2d(input_dim, output_dim, kernel_size=patch_size, stride=patch_size),
			Rearrange('b c h w -> b (h w) c'),
			nn.LayerNorm(output_dim),
		)

	def forward(self, x):
		(B, L, D) = x.shape
		assert L == self.input_size[0] * self.input_size[1] and D == self.input_dim
		return self.pooling(x)