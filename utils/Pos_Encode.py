import torch
from math import sin, cos
from einops import rearrange
from timm.models.layers import to_2tuple

def get_2dPE_matrix(h_dim, w_dim, channel, dev=None):
	''' Get 2D Positional Encoding Matrix. '''
	assert channel%4 == 0
	d = channel // 2
	PE = torch.empty((h_dim, w_dim, channel))
	PE = PE.to(dev) if dev is not None else PE
	for w_pos in range(h_dim):
		for h_pos in range(w_dim):
			hw_PE = []
			for i in range(channel//4):
				hw_PE.append(sin(h_pos/(10000**(2*i/d))))
				hw_PE.append(cos(h_pos/(10000**(2*i/d))))
			for i in range(channel//4):
				hw_PE.append(sin(w_pos/(10000**(2*i/d))))
				hw_PE.append(cos(w_pos/(10000**(2*i/d))))
			PE[h_pos, w_pos] = torch.tensor(hw_PE)
	PE = rearrange(PE, 'h w embed_dim -> (h w) embed_dim')
	return PE
			
def get_relative_position_index(matrix_size):
	matrix_h, matrix_w = to_2tuple(matrix_size)

	coords_h = torch.arange(matrix_h)
	coords_w = torch.arange(matrix_w)
	coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
	coords_flatten = torch.flatten(coords, 1)
	relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
	relative_coords = relative_coords.permute(1, 2, 0).contiguous()
	relative_coords[:, :, 0] += matrix_h - 1
	relative_coords[:, :, 1] += matrix_w - 1
	relative_coords[:, :, 0] *= 2 * matrix_w - 1
	relative_position_index = relative_coords.sum(-1)

	return relative_position_index