from model.SelfAttention import SelfAttention
from test.dataset import RandomDataset1
import torch
from torch import nn
from torch.utils.data import DataLoader

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		embed_num = 24
		self.net1 = nn.Sequential(
			SelfAttention(embed_num, 32, 24),
			SelfAttention(embed_num, 24, 16),
			SelfAttention(embed_num, 16, 8),
			SelfAttention(embed_num, 8, 4),
			SelfAttention(embed_num, 4, 1)
		)
	
	def forward(self, x):
		(bs, h, w) = x.shape
		x = self.net1(x)
		x = x.view(bs, w)
		return x


net = Net().cuda()
dataset = RandomDataset1(32, 24, 1)
dataloader = DataLoader(dataset, batch_size=128)
l = nn.L1Loss()
opt = torch.optim.Adam(net.parameters(), lr=3e-4)

def train():
	running_loss = 0.0
	for epoch, d in enumerate(dataloader):
		(x, y) = d
		x = x.cuda()
		y = y.cuda()

		opt.zero_grad()
		y_pred = net(x)
		#print(y.shape)
		#print(y_pred.shape)
		loss = l(y_pred, y)
		loss.backward()
		opt.step()

		running_loss += loss.item()
	print(f"loss = {running_loss}")