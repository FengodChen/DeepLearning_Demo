from torch.utils.data.dataloader import DataLoader
from data.Datasets import cifar_dataset, mnist_dataset
from model.ViT_Pytorch import ViT
from data.Trainer import Trainer
from data.Logger import ViT_Logger
import torch

dev = torch.device("cuda:0")
#dev = torch.device("cpu")
dataset = cifar_dataset("datasets", True)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
net = ViT(image_size=32, patch_size=2, dim=16, num_classes=10, channels=3, heads=16, depth=6, mlp_dim=16, dim_head=16).to(dev)
loss = torch.nn.CrossEntropyLoss()
opt = torch.optim.Adam(net.parameters(), lr=3e-4)
logger = ViT_Logger("save/test2", net)

trainer = Trainer(net, dataloader, loss, opt, dev, logger, 1)