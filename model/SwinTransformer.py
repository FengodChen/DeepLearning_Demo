import torch
from torch import nn, einsum
from math import sqrt
from einops.layers.torch import Rearrange
from einops import rearrange
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from utils.Pos_Encode import get_relative_position_index
from utils.Mask import get_sw_mask
from utils.Window import window_encode, window_decode

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
		)

	def forward(self, x):
		(B, C, H, W) = x.shape
		assert H == self.img_size[0] and W == self.img_size[1]
		return self.embed_enc(x)
	
	def get_io_size(self):
		img_h, img_w = self.img_size
		patch_h, patch_w = self.patch_size
		patch_h_num, patch_w_num = img_h // patch_h, img_w // patch_w

		input_size = self.img_size
		output_size = (patch_h_num, patch_w_num)
		return (input_size, output_size)

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
			Rearrange('b (h w) c -> b c h w', h=input_h),
			nn.Conv2d(input_dim, output_dim, kernel_size=patch_size, stride=patch_size),
			Rearrange('b c h w -> b (h w) c'),
		)

	def forward(self, x):
		(B, L, D) = x.shape
		assert L == self.input_size[0] * self.input_size[1] and D == self.input_dim
		return self.pooling(x)

class Classifier(nn.Module):
	def __init__(self, input_size, input_dim, classes_num):
		super().__init__()
		input_h, input_w = to_2tuple(input_size)
		self.input_num = input_h * input_w
		self.input_dim = input_dim

		self.dec_transform = nn.Linear(self.input_num, 1)
		self.dec = nn.Sequential(
		    nn.LayerNorm(input_dim),
		    nn.Linear(input_dim, classes_num)
		)
	
	def forward(self, x):
		B, L, D = x.shape
		assert L == self.input_num and D == self.input_dim

		x = (self.dec_transform(x.transpose(1, 2))).view(B, D)
		x = self.dec(x)
		return x

class WindowAttention(nn.Module):
	def __init__(self, input_size, heads_num, window_size, shift_size, input_dim, output_dim=None, qkv_bias=True, dropout=0.0):
		super().__init__()
		assert input_dim % heads_num == 0
		heads_dim = input_dim // heads_num
		output_dim = output_dim if output_dim is not None else input_dim
		self.heads_dim = heads_dim
		self.heads_num = heads_num
		self.input_dim = input_dim
		self.output_dim = output_dim
		window_h, window_w = self.window_size = to_2tuple(window_size)
		self.shift_size = to_2tuple(shift_size)
		self.input_size = to_2tuple(input_size)
		self.embed_num = window_h * window_w
		
		pos_dim = (2 * window_h - 1) * (2 * window_w - 1)
		relative_position_index = get_relative_position_index(window_size).view(-1)
		self.register_buffer("relative_position_index", relative_position_index)
		self.relative_position_table = nn.Parameter(torch.empty(pos_dim, heads_num))
		trunc_normal_(self.relative_position_table, std=.02)

		self.W_qkv = nn.Linear(input_dim, heads_dim*heads_num*3, bias=qkv_bias)
		self.div_qkv = Rearrange('b embed_num (qkv heads_num heads_dim) -> qkv b heads_num embed_num heads_dim', heads_num=heads_num, qkv=3)
		self.reshape_pe = Rearrange("(attn_i attn_j) heads_num -> heads_num attn_i attn_j", attn_i=window_h*window_w)
		self.softmax = nn.Softmax(dim=-1)
		self.combine = Rearrange('b heads_num embed_num inner_dim -> b embed_num (heads_num inner_dim)', heads_num=heads_num)
		self.out = nn.Sequential(
		    nn.Linear(heads_dim*heads_num, output_dim), 
		    nn.Dropout(dropout)
		)

	def forward(self, x, mask=None):
		x = window_encode(x, self.input_size, self.window_size, self.shift_size)

		BW, N, D = x.shape
		window_h, window_w = self.window_size
		assert N == window_h * window_w and D == self.input_dim

		(q, k, v) = self.div_qkv(self.W_qkv(x))
		a = einsum('bnid, bnjd -> bnij', k, q) / sqrt(self.heads_dim)

		relative_position_bias = self.reshape_pe(self.relative_position_table[self.relative_position_index])
		a = a + relative_position_bias

		if mask is not None:
			nWindow = mask.shape[1]
			a = rearrange(a, "(b nWindow) nHeads attn_i attn_j -> b nWindow nHeads attn_i attn_j", nWindow=nWindow)
			a = a + mask
			a = rearrange(a, "b nWindow nHeads attn_i attn_j -> (b nWindow) nHeads attn_i attn_j", nWindow=nWindow)

		a = self.softmax(a)

		y = einsum('bnij,bnjk -> bnik', a, v)
		y = self.out(self.combine(y))

		y = window_decode(y, self.input_size, self.window_size, self.shift_size)

		return y

class Mlp(nn.Module):
	def __init__(self, input_dim, hidden_ratio=None, output_dim=None, dropout=0.0):
		super().__init__()
		hidden_dim = int(input_dim * hidden_ratio) if hidden_ratio is not None else input_dim
		output_dim = output_dim if output_dim is not None else input_dim

		self.mlp = nn.Sequential(
			nn.Linear(input_dim, hidden_dim),
			nn.GELU(),
			nn.Dropout(dropout),
			nn.Linear(hidden_dim, output_dim),
			nn.Dropout(dropout)
		)
	
	def forward(self, x):
		return self.mlp(x)

class SwinTransformerBlock(nn.Module):
	def __init__(self, input_size, input_dim, window_size, shift_size, heads_num,
					output_dim=None, wsa_output_dim=None, mlp_inner_ratio=None,
					wsa_dropout=0.0, mlp_dropout=0.0, droppath=0.0,
					qkv_bias=True):
		super().__init__()

		wsa_output_dim = wsa_output_dim if wsa_output_dim is not None else input_dim

		self.LN1 = nn.LayerNorm(input_dim)
		self.WSA = WindowAttention(
			input_size = input_size,
			heads_num = heads_num,
			window_size = window_size,
			shift_size = shift_size,
			input_dim = input_dim,
			output_dim = wsa_output_dim,
			qkv_bias = qkv_bias,
			dropout = wsa_dropout
		)
		self.LN2 = nn.LayerNorm(wsa_output_dim)
		self.MLP = Mlp(
			input_dim = wsa_output_dim,
			hidden_ratio = mlp_inner_ratio,
			output_dim = output_dim,
			dropout = mlp_dropout
		)
		self.DropPath = DropPath(droppath)

		mask = get_sw_mask(input_size, window_size, shift_size)
		self.register_buffer("mask", mask)
	
	def forward(self, x):
		x = x + self.DropPath(self.WSA(self.LN1(x), self.mask))
		x = x + self.DropPath(self.MLP(self.LN2(x)))

		return x

class BasicBlock(nn.Module):
	def __init__(self, input_size, input_dim, window_size, heads_num, depth_num, pre_pooling:bool,
					mlp_inner_ratio=None,
					wsa_dropout=0.2, mlp_dropout=0.2, droppath=0.2,
					qkv_bias=True):
		super().__init__()
		input_h, input_w = self.input_size = to_2tuple(input_size)
		self.pre_pooling = pre_pooling

		window_h, window_w = to_2tuple(window_size)

		inner_h = input_h // 2 if pre_pooling else input_h
		inner_w = input_w // 2 if pre_pooling else input_w

		window_shift_h = window_h // 2 if window_h >= inner_h else 0
		window_shift_w = window_w // 2 if window_w >= inner_w else 0
		window_h = inner_h if window_h > inner_h else window_h
		window_w = inner_w if window_w > inner_w else window_w

		window_size = (window_h, window_w)

		pooling_size = (inner_h, inner_w) if pre_pooling else input_size
		pooling_dim = input_dim * 2 if pre_pooling else input_dim

		self.pooling = Embed_Pooling(input_size, pooling_size, input_dim, pooling_dim) if pre_pooling else nn.Identity()
		self.wsa_list = nn.ModuleList()

		for i in range(depth_num):
			shift_size = (window_shift_h, window_shift_w) if i % 2 != 0 else 0
			wsa = SwinTransformerBlock(
				input_size = pooling_size,
				input_dim = pooling_dim,
				window_size = window_size,
				shift_size = shift_size,
				heads_num = heads_num,
				mlp_inner_ratio = mlp_inner_ratio,
				wsa_dropout = wsa_dropout, mlp_dropout = mlp_dropout, droppath = droppath,
				qkv_bias = qkv_bias
			)
			self.wsa_list.append(wsa)
	
	def forward(self, x):
		x = self.pooling(x)
		for wsa in self.wsa_list:
			x = wsa(x)
		return x
	
	def get_io_size(self):
		input_h, input_w = self.input_size

		input_size = self.input_size
		output_size = (input_h // 2, input_w // 2) if self.pre_pooling else self.input_size

		return (input_size, output_size)


class SwinTransformer(nn.Module):
	def __init__(self, image_size, image_channel, patch_size, embed_dim, window_size, classes_num, mlp_inner_ratio,
					heads_num_list, depth_num_list,
					wsa_dropout=0.2, mlp_dropout=0.2, droppath=0.2,
					qkv_bias=True):
		super().__init__()

		assert len(heads_num_list) == len(depth_num_list)

		self.image_size = to_2tuple(image_size)
		self.image_channel = image_channel
		
		# Patch Embedding
		self.embed_coder = Embed_Encoder(
			img_size = image_size,
			patch_size = patch_size,
			img_channel = image_channel,
			embed_dim = embed_dim
		)
		_, inner_size = self.embed_coder.get_io_size()
		inner_dim = embed_dim

		# Forward Features
		self.blocks_list = nn.ModuleList()
		blocks_num = len(depth_num_list)
		for ptr in range(blocks_num):
			pre_pooling = True if ptr > 0 else False
			block = BasicBlock(
				input_size = inner_size,
				input_dim = inner_dim,
				window_size = window_size,
				heads_num = heads_num_list[ptr],
				depth_num = depth_num_list[ptr],
				pre_pooling = pre_pooling,
				mlp_inner_ratio = mlp_inner_ratio,
				wsa_dropout = wsa_dropout,
				mlp_dropout = mlp_dropout,
				droppath = droppath,
				qkv_bias = qkv_bias
			)
			_, inner_size = block.get_io_size()
			inner_dim = inner_dim * 2 if ptr > 0 else inner_dim
			self.blocks_list.append(block)
		
		# Classifier
		self.classifier = Classifier(
			input_size = inner_size,
			input_dim = inner_dim,
			classes_num = classes_num
		)
	
	def forward(self, x):
		(B, C, H, W) = x.shape
		assert C == self.image_channel and H == self.image_size[0] and W == self.image_size[1]

		x = self.embed_coder(x)
		for block in self.blocks_list:
			x = block(x)
		x = self.classifier(x)

		return x
