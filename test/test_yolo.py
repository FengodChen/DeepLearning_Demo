from torch.utils.data.dataloader import DataLoader
from data.Datasets import cifar_dataset, mnist_dataset, voc_dataset
from utils.Compare_Func import cifar10_compare_func as compare_func
from utils.Seed import seed_everything
from data.Trainer import Trainer
from data.Logger import Logger
from model.Yolo3 import Yolo3
import torch

torch.backends.cudnn.benchmark = True

seed_everything(21120009)

CLASSES_NUM = 80
ANCHOR_NUM = 3
GPU_NUM = 2

dataset_train = voc_dataset(root_dir="datasets", year="2007", image_set="train")
dataloader_train = DataLoader(dataset_train, batch_size=64, shuffle=True)
dataset_val = voc_dataset(root_dir="datasets", year="2007", image_set="val")
dataloader_val = DataLoader(dataset_val, batch_size=64, shuffle=True)
dataset_test = voc_dataset(root_dir="datasets", year="2007", image_set="test")
dataloader_test = DataLoader(dataset_test, batch_size=64, shuffle=True)

dev = torch.device("cuda")
net = Yolo3(in_channel=3, classes_num=CLASSES_NUM, anchor_num=ANCHOR_NUM, gpu_num=GPU_NUM).to(dev)

x = torch.rand(2, 3, 256, 256).to(dev)
