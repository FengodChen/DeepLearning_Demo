from torch.utils.data.dataloader import DataLoader
from data.Datasets import Voc_Dataset
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
GPU_NUM = 0
IMG_SIZE = (224, 224)

dataset_train = Voc_Dataset(root_dir="datasets", year="2007", resize=IMG_SIZE, image_set="train")
dataloader_train = DataLoader(dataset_train, batch_size=64, shuffle=True)
dataset_val = Voc_Dataset(root_dir="datasets", year="2007", resize=IMG_SIZE, image_set="val")
dataloader_val = DataLoader(dataset_val, batch_size=64, shuffle=True)
#dataset_test = Voc_Dataset(root_dir="datasets", year="2007", resize=IMG_SIZE, image_set="test")
#dataloader_test = DataLoader(dataset_test, batch_size=64, shuffle=True)

dev = torch.device("cpu")
net = Yolo3(in_channel=3, classes_num=CLASSES_NUM, anchor_num=ANCHOR_NUM, gpu_num=GPU_NUM).to(dev)

