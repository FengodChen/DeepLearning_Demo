import torch
from einops import rearrange
from timm.models.layers import to_2tuple

def window_encode(x, input_size, window_size, shift_size):
    input_h, input_w = input_size = to_2tuple(input_size)
    window_h, window_w = window_size = to_2tuple(window_size)
    shift_h, shift_w = shift_size = to_2tuple(shift_size)
    B, L, C = x.shape

    assert input_h % window_h == 0 and input_w % window_w == 0
    assert L == input_h * input_w

    twoDim_x = rearrange(x, "b (h w) c -> b h w c", h=input_h)

    if shift_h == 0 and shift_w == 0:
        shifted_x = twoDim_x
    else:
        shifted_x = torch.roll(twoDim_x, shifts=(-shift_h, -shift_w), dims=(1, 2))
    
    window_x = rearrange(shifted_x, "B (nH H) (nW W) C -> (B nH nW) (H W) C", H=window_h, W=window_w)

    return window_x

def window_decode(window_x, input_size, window_size, shift_size):
    '''
    input_size, window_size and shift_size same as window_encode()
    '''
    input_h, input_w = input_size = to_2tuple(input_size)
    window_h, window_w = window_size = to_2tuple(window_size)
    shift_h, shift_w = shift_size = to_2tuple(shift_size)
    BW, L, C = window_x.shape

    assert input_h % window_h == 0 and input_w % window_w == 0
    assert L == window_h * window_w

    window_h_num = input_h // window_h
    window_w_num = input_w // window_w

    shifted_x = rearrange(window_x, "(B nH nW) (H W) C -> B (nH H) (nW W) C", H=window_h, nH=window_h_num, nW=window_w_num)

    if shift_h == 0 and shift_w == 0:
        twoDim_x = shifted_x
    else:
        twoDim_x = torch.roll(shifted_x, shifts=(shift_h, shift_w), dims=(1, 2))

    x = rearrange(twoDim_x, "b h w c -> b (h w) c", h=input_h)

    return x
