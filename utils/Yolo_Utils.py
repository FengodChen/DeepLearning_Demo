import pickle
from torch.utils.data.dataloader import DataLoader
import torch
import random

class Voc_Kmeans:
    def __init__(self, voc_dataset, load_path=None, cluster_num=None):
        self.voc_dataset = voc_dataset
        self.cluster_list = None

        if load_path is None:
            assert cluster_num is not None
            self.generate_cluster(cluster_num)
        else:
            self.load(load_path)

    def generate_cluster(self, cluster_num):
        bndboxs = self.get_all_bndbox()
        self.kmeans(bndboxs, cluster_num)
    
    def get_all_bndbox(self):
        bndboxs = []
        for i, d in enumerate(self.voc_dataset):
            objs = d[1]["annotation"]["object"]
            h = float(d[1]["annotation"]["size"]["height"])
            w = float(d[1]["annotation"]["size"]["width"])
            for obj in objs:
                bndbox = obj["bndbox"]
                bndbox_h = int(bndbox["ymax"]) - int(bndbox["ymin"])
                bndbox_w = int(bndbox["xmax"]) - int(bndbox["xmin"])

                bndbox_h = bndbox_h / h
                bndbox_w = bndbox_w / w
                bndboxs.append([bndbox_h, bndbox_w])
            print(f"[reading] {i+1}/{len(self.voc_dataset)}")
        bndboxs = torch.tensor(bndboxs)
        return bndboxs
    
    def get_iou(self, cluster, bndboxs):
        cluster_area = cluster[0] * cluster[1]
        bndboxs_area = bndboxs[:, 0] * bndboxs[:, 1]
        intersection = torch.minimum(cluster, bndboxs)
        intersection_area = intersection[:, 0] * intersection[:, 1]

        iou = intersection_area / (cluster_area + bndboxs_area - intersection_area)
        iou = iou.view(-1, 1)
        return iou
    
    def get_avg_iou(self, cluster, bndboxs):
        iou = self.get_iou(cluster, bndboxs)
        s = iou.sum()
        s = s / bndboxs.shape[0]
        return s
    
    def kmeans(self, bndboxs, cluster_num, eps=1e-12):
        # Initialize cluster
        (bndboxs_num, dim) = bndboxs.shape
        self.cluster_list = torch.zeros((cluster_num, 2))
        index_list = [i for i in range(bndboxs_num)]
        random.shuffle(index_list)
        index_list = index_list[:cluster_num]
        for i, index in enumerate(index_list):
            self.cluster_list[i] = bndboxs[index]
        
        last_iou_avg = 0.0
        while True:
            # Bound box find nearest cluster
            distance = torch.zeros((bndboxs_num, 0))
            for cluster in self.cluster_list:
                d = 1 - self.get_iou(cluster, bndboxs)
                distance = torch.concat((distance, d), dim=-1)
            distance_argmin = torch.argmin(distance, dim=-1)

            # Generalize new cluster
            iou_sum = 0.0
            for i in range(cluster_num):
                bndboxs_n = bndboxs[distance_argmin == i].to(torch.float)
                cluster = torch.mean(bndboxs_n, dim=0)
                self.cluster_list[i] = cluster
                iou_sum = iou_sum + self.get_avg_iou(cluster, bndboxs_n)
            iou_avg = iou_sum / float(cluster_num)
            
            # Decision if break loop
            delta = torch.abs(last_iou_avg - iou_avg)
            if delta <= eps:
                break
            last_iou_avg = torch.clone(iou_avg)

            # Show
            print(f"[avg iou]: {iou_avg}")
            print(f"[delta]: {delta}")

    def load(self, file_path):
        with open(file_path, "rb") as f:
            self.cluster_list = pickle.load(f)
    
    def save(self, file_path):
        with open(file_path, "wb") as f:
            pickle.dump(self.cluster_list, f)


class Voc_Dataset_Prepare:
    def __init__(self, voc_dataset, load_path=None):
        self.voc_dataset = voc_dataset
        self.label_name = []

        if load_path is None:
            self.generate_label()
        else:
            self.load(load_path)

    def generate_label(self):
        for i, d in enumerate(self.voc_dataset):
            objs = d[1]["annotation"]["object"]
            for obj in objs:
                label_name = obj["name"]
                if not label_name in self.label_name:
                    self.label_name.append(label_name)
            print(f"[reading] {i+1}/{len(self.voc_dataset)}")
    
    def label_name2num(self, name):
        num = self.label_name.index(name)
        return num
    
    def label_name2onehot(self, name):
        num = self.label_name2num(name)
        classes_num = len(self.label_name)

        onehot = torch.zeros(classes_num)
        onehot[num] = 1

        return onehot
    
    def load(self, file_path):
        with open(file_path, "rb") as f:
            self.label_name = pickle.load(f)
    
    def save(self, file_path):
        with open(file_path, "wb") as f:
            pickle.dump(self.label_name, f)
