import torch
from math import sin, cos

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
	return PE
			
