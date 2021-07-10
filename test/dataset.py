import torch
from torch import random
from torch.utils.data import Dataset

class RandomDataset1(Dataset):
    def __init__(self, x_dim, x_num, y_dim, len=1024):
        self.x_dim = x_dim
        self.x_num = x_num
        self.y_dim = y_dim

        self.len = len

    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        x = torch.rand(self.x_dim, self.x_num)
        y = torch.sum(x, dim=0) / self.x_dim
        return (x, y)

