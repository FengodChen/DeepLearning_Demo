from matplotlib import pyplot
from torch.utils.data.dataloader import DataLoader
from data.Datasets import Voc_Dataset
from model.Loss import YOLO3_Loss
from utils.Compare_Func import cifar10_compare_func as compare_func, void_compare_func
from utils.Seed import seed_everything
from data.Trainer import YOLO3_Trainer
from data.Logger import Logger
from model.Yolo3 import Yolo3
import torch
import matplotlib.pyplot as plt

from utils.Yolo_Utils import VOC_Utils, Voc_Dataset_Prepare, Voc_Kmeans, collate_fn

torch.backends.cudnn.benchmark = True

seed_everything(21120009)

BATCH_SIZE = 4
CLUSTER_NUM = 9
ANCHOR_NUM = 3
GPU_NUM = -1
IMG_SIZE = (224, 224)
VOC_KMEANS_PATH = f"datasets/voc_kmeans-cluster_num_{CLUSTER_NUM}.pkl"
VOC_DATASET_PREPARE_PATH = "datasets/voc_dataset_prepare.pkl"
LAMBDA_COORD = 10
LAMBDA_OBJ = 5
LAMBDA_NOOBJ = 5
LAMBDA_CLS = 1

dataset_train = Voc_Dataset(root_dir="datasets", year="2007", resize=IMG_SIZE, image_set="train")
dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
dataset_val = Voc_Dataset(root_dir="datasets", year="2007", resize=IMG_SIZE, image_set="val")
dataloader_val = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
#dataset_test = Voc_Dataset(root_dir="datasets", year="2007", resize=IMG_SIZE, image_set="test")
#dataloader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

dataset_train_show = Voc_Dataset(root_dir="datasets", year="2007", resize=IMG_SIZE, image_set="train", for_show=True)
dataset_val_show = Voc_Dataset(root_dir="datasets", year="2007", resize=IMG_SIZE, image_set="val", for_show=True)

try:
    voc_kmeans = Voc_Kmeans(dataset_train, load_path=VOC_KMEANS_PATH)
except:
    voc_kmeans = Voc_Kmeans(dataset_train, cluster_num=CLUSTER_NUM)
    voc_kmeans.save(VOC_KMEANS_PATH)
try:
    voc_dataset_prepare = Voc_Dataset_Prepare(dataset_train, load_path=VOC_DATASET_PREPARE_PATH)
except:
    voc_dataset_prepare = Voc_Dataset_Prepare(dataset_train)
    voc_dataset_prepare.save(VOC_DATASET_PREPARE_PATH)


CLASSES_NUM = len(voc_dataset_prepare.label_name)

dev = torch.device("cpu") if GPU_NUM < 0 else torch.device(f"cuda:{GPU_NUM}")
net = Yolo3(in_channel=3, classes_num=CLASSES_NUM, anchor_num=ANCHOR_NUM).to(dev)

voc_utils = VOC_Utils(IMG_SIZE, voc_kmeans, voc_dataset_prepare, net, ANCHOR_NUM, dev)

dataset_train.set_voc_util(voc_utils)
dataset_val.set_voc_util(voc_utils)
dataset_train_show.set_voc_util(voc_utils)
dataset_val_show.set_voc_util(voc_utils)
#dataset_test.set_voc_util(voc_utils)
#dataset_test_show.set_voc_util(voc_utils)

net.set_end_func(voc_utils.yolo3_encode_to_tensor)

logger = Logger("save/yolo3", net, load_newest=True)
loss = YOLO3_Loss(voc_utils, lambda_coord=LAMBDA_COORD, lambda_obj=LAMBDA_OBJ, lambda_noobj=LAMBDA_NOOBJ, lambda_cls=LAMBDA_CLS)
opt = torch.optim.Adam(net.parameters(), lr=3e-4)

trainer = YOLO3_Trainer(net, loss, opt, dev, logger, 10)
trainer.train(dataloader_train, dataloader_val, void_compare_func, 10, 500)

logger.plot_log("train", "show")
logger.plot_log("eval", "show")

net.eval()

for (d, d_show) in zip(dataset_val, dataset_val_show):
    x = d[0]
    y_true = d[1:]
    x_show = d_show[0]

    x = x.unsqueeze(0)
    y = net(x)
    y_true = [i.unsqueeze(0) for i in y_true]

    x = (x_show * 255).to(torch.uint8)

    y_true_draw = voc_utils.draw_bbox(x, y_true, 0.9)
    y_pred_draw = voc_utils.draw_bbox(x, y, 0.9)

    x = x.permute(1, 2, 0).detach().cpu().numpy()
    y_pred_draw = y_pred_draw.permute(1, 2, 0).detach().cpu().numpy()
    y_true_draw = y_true_draw.permute(1, 2, 0).detach().cpu().numpy()

    plt.figure(figsize=(24, 12))

    plt.subplot(1, 3, 1)
    plt.imshow(x)
    plt.title("x")

    plt.subplot(1, 3, 2)
    plt.imshow(y_true_draw)
    plt.title("y true")

    plt.subplot(1, 3, 3)
    plt.imshow(y_pred_draw)
    plt.title("y pred")

    plt.show()
