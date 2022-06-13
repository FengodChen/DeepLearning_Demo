from torch.utils.data.dataloader import DataLoader
from data.Datasets import Voc_Dataset
from utils.Compare_Func import cifar10_compare_func as compare_func
from utils.Seed import seed_everything
from data.Trainer import Trainer
from data.Logger import Logger
from model.Yolo3 import Yolo3
import torch

from utils.Yolo_Utils import VOC_Utils, Voc_Dataset_Prepare, Voc_Kmeans

torch.backends.cudnn.benchmark = True

seed_everything(21120009)

CLUSTER_NUM = 9
ANCHOR_NUM = 3
GPU_NUM = 0
IMG_SIZE = (224, 224)
VOC_KMEANS_PATH = f"datasets/voc_kmeans-cluster_num_{CLUSTER_NUM}.pkl"
VOC_DATASET_PREPARE_PATH = "datasets/voc_dataset_prepare.pkl"

dataset_train = Voc_Dataset(root_dir="datasets", year="2007", resize=IMG_SIZE, image_set="train")
dataloader_train = DataLoader(dataset_train, batch_size=64, shuffle=True)
dataset_val = Voc_Dataset(root_dir="datasets", year="2007", resize=IMG_SIZE, image_set="val")
dataloader_val = DataLoader(dataset_val, batch_size=64, shuffle=True)
#dataset_test = Voc_Dataset(root_dir="datasets", year="2007", resize=IMG_SIZE, image_set="test")
#dataloader_test = DataLoader(dataset_test, batch_size=64, shuffle=True)

try:
    voc_kmeans = Voc_Kmeans(dataset_train, load_path=VOC_KMEANS_PATH)
except:
    voc_kmeans = Voc_Kmeans(dataset_train, cluster_num=CLUSTER_NUM)
try:
    voc_dataset_prepare = Voc_Dataset_Prepare(dataset_train, load_path=VOC_DATASET_PREPARE_PATH)
except:
    voc_dataset_prepare = Voc_Dataset_Prepare(dataset_train)

voc_utils = VOC_Utils(voc_kmeans, voc_dataset_prepare, ANCHOR_NUM)

CLASSES_NUM = len(voc_dataset_prepare.label_name)

if GPU_NUM == 0:
    dev = torch.device("cpu")
    net = Yolo3(in_channel=3, classes_num=CLASSES_NUM, anchor_num=ANCHOR_NUM, gpu_num=GPU_NUM).to(dev)
elif GPU_NUM == 1:
    dev = torch.device("cuda")
    net = Yolo3(in_channel=3, classes_num=CLASSES_NUM, anchor_num=ANCHOR_NUM, gpu_num=GPU_NUM).to(dev)
elif GPU_NUM > 1:
    dev = torch.device("cuda")
    net = Yolo3(in_channel=3, classes_num=CLASSES_NUM, anchor_num=ANCHOR_NUM, gpu_num=GPU_NUM)
    net = torch.nn.DataParallel(net).to(dev)


logger = Logger("save/yolo3", net, load_newest=True)
loss = torch.nn.CrossEntropyLoss()
opt = torch.optim.Adam(net.parameters(), lr=3e-4)

trainer = Trainer(net, loss, opt, dev, logger, 10)