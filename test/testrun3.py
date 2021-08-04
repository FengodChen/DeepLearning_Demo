from torch.utils.data.dataloader import DataLoader
from data.Datasets import cifar_dataset, mnist_dataset
from model.SelfAttention import ViT
from data.Trainer import Trainer
from data.Logger import ViT_Logger
import torch

dev = torch.device("cuda:0")
#dev = torch.device("cpu")
dataset = cifar_dataset("datasets", True)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
net = ViT((32, 32), (2, 2), 16, 10, img_channel=3, dev=dev, sa_num=16, msa_num=6).to(dev)
loss = torch.nn.CrossEntropyLoss()
opt = torch.optim.Adam(net.parameters(), lr=3e-4)
logger = ViT_Logger("save/test1", net)

trainer = Trainer(net, dataloader, loss, opt, dev, logger, 1)