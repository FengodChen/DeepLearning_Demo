from torch.utils.data.dataloader import DataLoader
from data.Datasets import cifar_dataset, mnist_dataset
from model.SelfAttention import ViT
#from model.SelfAttention_Test import ViT
from data.Trainer import Trainer
from data.Logger import ViT_Logger
import torch

def compare_func(y_pred, y):
	y_pred_ = torch.argmax(y_pred, dim=1).view(-1)
	y_ = y.view(-1)
	true_ans = y_-y_pred_
	true_ans = true_ans[true_ans == 0]
	acc = len(true_ans) / len(y_)
	return acc

dev = torch.device("cuda:0")
#dev = torch.device("cpu")
dataset_train = cifar_dataset("datasets", True)
dataloader_train = DataLoader(dataset_train, batch_size=16, shuffle=True)
dataset_test = cifar_dataset("datasets", True, train=False)
dataloader_test = DataLoader(dataset_test, batch_size=8, shuffle=True)
net = ViT((32, 32), (2, 2), 16, 10, img_channel=3, dev=dev, heads_num=8, msa_num=4).to(dev)
loss = torch.nn.CrossEntropyLoss()
opt = torch.optim.Adam(net.parameters(), lr=3e-4)
logger = ViT_Logger("save/ViT_True_MSA", net, load_newest=True)

trainer = Trainer(net, loss, opt, dev, logger, 1)