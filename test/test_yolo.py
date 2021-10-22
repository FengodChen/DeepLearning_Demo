from torch.utils.data.dataloader import DataLoader
from data.Datasets import cifar_dataset, mnist_dataset
from utils.Compare_Func import cifar10_compare_func as compare_func
from utils.Seed import seed_everything
from data.Trainer import Trainer
from data.Logger import Logger
from model.Yolo3 import Yolo3
import torch

torch.backends.cudnn.benchmark = True

seed_everything(17211401)

dev = torch.device("cuda")
net = Yolo3(in_channel=3, classes_num=80, anchor_num=3)
