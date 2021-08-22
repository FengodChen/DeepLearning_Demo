import torch
from einops import rearrange
from timm.models.layers import to_2tuple

def get_sw_mask(input_size, window_size, shift_size):
	input_h, input_w = to_2tuple(input_size)
	window_h, window_w = to_2tuple(window_size)
	shift_h, shift_w = to_2tuple(shift_size)
	if shift_h == 0 and shift_w == 0:
		return None

	img_mask = torch.zeros((1, input_h, input_w, 1))
	h_slices = (slice(0, -window_h), slice(-window_h, -shift_h), slice(-shift_h, None))
	w_slices = (slice(0, -window_w), slice(-window_w, -shift_w), slice(-shift_w, None))

	cnt = 0
	for h in h_slices:
		for w in w_slices:
			img_mask[:, h, w, :] = cnt
			cnt += 1

	mask_windows = rearrange(img_mask, "B (nH H) (nW W) D -> (B nH nW) (H W D)", H=window_h, W=window_w)
	attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
	attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

	return attn_mask.unsqueeze(1).unsqueeze(0)