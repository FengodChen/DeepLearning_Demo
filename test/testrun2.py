from torch.utils.data.dataloader import DataLoader
from data.Datasets import cifar_dataset, mnist_dataset
from model.SelfAttention import ViT
import torch

dev = torch.device("cuda:0")
#dev = torch.device("cpu")
dataset = cifar_dataset("datasets", True)
dataloader = DataLoader(dataset, batch_size=2)
net = ViT((32, 32), (2, 2), 16, 10, img_channel=3, dev=dev, heads_num=16, msa_num=6).to(dev)
l = torch.nn.CrossEntropyLoss()
opt = torch.optim.Adam(net.parameters(), lr=3e-4)

def train():
	net.train()
	opt_epoch = 32
	running_loss = 0.0
	for epoch, d in enumerate(dataloader):
		(x, y) = d
		x = x.to(dev)
		y = y.to(dev)

		y_pred = net(x)
		loss = l(y_pred, y)
		loss.backward()
		opt.step()
		opt.zero_grad()

		if (epoch % opt_epoch == 0):
			print(f"epoch = {epoch}, loss = {running_loss/opt_epoch}")
			running_loss = 0.0

		running_loss += loss.item()

def eval(model_path, batch_size=2, log_path=None):
	dataset = cifar_dataset("datasets", download=True, train=False)
	dataloader = DataLoader(dataset, batch_size=batch_size)
	net.load_state_dict(torch.load(model_path))
	net.eval()
	trueAns_num = 0
	for epoch, d in enumerate(dataloader):
		(x, y) = d
		x = x.to(dev)
		y = y.to(dev)
		y_pred = net(x)
		y_pred_fix = torch.argmax(y_pred, dim=1)
		compare_matrix = y_pred_fix - y
		trueAns_num += len(compare_matrix[compare_matrix == 0])
		print(f"{(epoch+1)*batch_size}/{len(dataset)}")
	d = f"True Ans: {trueAns_num}, Total Ans: {len(dataset)}, Rate: {trueAns_num / len(dataset)}\n"
	print(d)
	if (log_path is not None):
		with open(log_path, "a") as f:
			f.write(d)
