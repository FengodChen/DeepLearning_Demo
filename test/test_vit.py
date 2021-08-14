from torch.utils.data.dataloader import DataLoader
from data.Datasets import cifar_dataset, mnist_dataset
from utils.Compare_Func import cifar10_compare_func as compare_func
from utils.Seed import seed_everything
from data.Trainer import Trainer
from data.Logger import ViT_Logger
import torch

seed_everything(17211401)

dev = torch.device("cuda:0")
dataset_train = cifar_dataset("datasets", True)
dataloader_train = DataLoader(dataset_train, batch_size=16, shuffle=True)
dataset_test = cifar_dataset("datasets", True, train=False)
dataloader_test = DataLoader(dataset_test, batch_size=8, shuffle=True)

'''
from model.SelfAttention import ViT
net = ViT(
	img_size = (32, 32),
	patch_size = (2, 2),
	embed_dim = 16, 
	classes = 10, 
	heads_num = 8, 
	msa_num = 4, 
	msa_inner_dim = 16,
	mlp_inner_dim = 16,
	dropout = 0.5,
	embed_dropout = 0.5,
	img_channel = 3,
	dev = dev,
	pos_learnable = False,
	qkv_bias = True
).to(dev)
logger = ViT_Logger("save/ViT_MSA_Pos_is_Learnable", net, load_newest=True)
#'''

#'''
from ref_model.ViT import ViT
net = ViT(
    image_size = 32,
    patch_size = 2,
    dim = 16,
    num_classes = 10,
    heads = 8,
    depth = 4,
    dim_head = 16,
    mlp_dim = 16,
    dropout = 0.1,
    emb_dropout = 0.1,
    channels = 3,
	dev = dev,
	pos_learnable= False,
	pool = 'linear'
).to(dev)
logger = ViT_Logger("save/Pytorch-ViT_Pos_is_Learnable", net, load_newest=True)
#'''

loss = torch.nn.CrossEntropyLoss()
opt = torch.optim.Adam(net.parameters(), lr=3e-4)

trainer = Trainer(net, loss, opt, dev, logger, 1)