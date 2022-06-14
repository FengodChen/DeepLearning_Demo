import pickle
from torch.utils.data.dataloader import DataLoader
import torch
import random
import math

class Voc_Kmeans:
    def __init__(self, voc_dataset, load_path=None, cluster_num=None):
        self.voc_dataset = voc_dataset
        self.cluster_list = None

        if load_path is None:
            assert cluster_num is not None
            self.generate_cluster(cluster_num)
        else:
            self.load(load_path)
    
    def sort(self):
        """
        Sort from small to big
        """
        s = self.cluster_list
        s = s[:, 0] * s[:, 1]
        ptr = torch.argsort(s)
        self.cluster_list = self.cluster_list[ptr]

    def generate_cluster(self, cluster_num):
        bboxs = self.get_all_bboxs()
        self.kmeans(bboxs, cluster_num)
    
    def get_all_bboxs(self):
        bboxs = []
        for i, d in enumerate(self.voc_dataset):
            objs = d[1]["annotation"]["object"]
            h = float(d[1]["annotation"]["size"]["height"])
            w = float(d[1]["annotation"]["size"]["width"])
            for obj in objs:
                bbox = obj["bndbox"]
                bbox_h = int(bbox["ymax"]) - int(bbox["ymin"])
                bbox_w = int(bbox["xmax"]) - int(bbox["xmin"])

                bbox_h = bbox_h / h
                bbox_w = bbox_w / w
                bboxs.append([bbox_h, bbox_w])
            print(f"[reading] {i+1}/{len(self.voc_dataset)}")
        bboxs = torch.tensor(bboxs)
        return bboxs
    
    def get_iou(self, cluster, bboxs):
        cluster_area = cluster[0] * cluster[1]
        bboxs_area = bboxs[:, 0] * bboxs[:, 1]
        intersection = torch.minimum(cluster, bboxs)
        intersection_area = intersection[:, 0] * intersection[:, 1]

        iou = intersection_area / (cluster_area + bboxs_area - intersection_area)
        iou = iou.view(-1, 1)
        return iou
    
    def get_avg_iou(self, cluster, bboxs):
        iou = self.get_iou(cluster, bboxs)
        s = iou.sum()
        s = s / bboxs.shape[0]
        return s
    
    def kmeans(self, bboxs, cluster_num, eps=1e-12):
        # Initialize cluster
        (bboxs_num, dim) = bboxs.shape
        self.cluster_list = torch.zeros((cluster_num, 2))
        index_list = [i for i in range(bboxs_num)]
        random.shuffle(index_list)
        index_list = index_list[:cluster_num]
        for i, index in enumerate(index_list):
            self.cluster_list[i] = bboxs[index]
        
        last_iou_avg = 0.0
        while True:
            # Bound box find nearest cluster
            distance = torch.zeros((bboxs_num, 0))
            for cluster in self.cluster_list:
                d = 1 - self.get_iou(cluster, bboxs)
                distance = torch.concat((distance, d), dim=-1)
            distance_argmin = torch.argmin(distance, dim=-1)

            # Generalize new cluster
            iou_sum = 0.0
            for i in range(cluster_num):
                bboxs_n = bboxs[distance_argmin == i].to(torch.float)
                cluster = torch.mean(bboxs_n, dim=0)
                self.cluster_list[i] = cluster
                iou_sum = iou_sum + self.get_avg_iou(cluster, bboxs_n)
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
        
        self.classes_num = len(self.label_name)

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

class VOC_Utils:
    def __init__(self, voc_kmeans:Voc_Kmeans, voc_dataset_prepare:Voc_Dataset_Prepare, anchor_num:int):
        assert len(voc_kmeans.cluster_list) % anchor_num == 0

        self.voc_kmeans = voc_kmeans
        self.voc_dataset_prepare = voc_dataset_prepare

        self.voc_kmeans.sort()
        self.anchor_list = self.voc_kmeans.cluster_list
        self.anchor_num = anchor_num
        self.classes_num = self.voc_dataset_prepare.classes_num
        
    def get_max_iou_arg(self, bbox, anchor):
        bbox_area = bbox[2] * bbox[3]
        anchor_area = anchor[:, 0] * anchor[:, 1]
        intersection = torch.minimum(anchor, bbox[2:4])
        intersection_area = intersection[:, 0] * intersection[:, 1]

        iou = intersection_area / (anchor_area + bbox_area - intersection_area)
        iou = iou.view(-1)
        return torch.argmax(iou)

    def decode_label(self, label):
        """
        ## args
        label from voc dataset
        ## return
        (img_size(H, W), bbox(N, (x_center, y_center, h, w))(normlized), classes(N, classes_num)(one-hot))
        """
        # Get Image Shape
        img_size = (img_h, img_w) = (int(label["annotation"]["size"]["height"]), int(label["annotation"]["size"]["width"]))

        # Get Classes and BoundBox
        # Classes: [One-Hot Encode]
        # BoundBox: [x_center, y_center, h, w]
        bbox_num = len(label["annotation"]["object"])
        classes_num = self.voc_dataset_prepare.classes_num

        bbox = torch.zeros((bbox_num, 4), dtype=torch.float)
        classes = torch.zeros((bbox_num, classes_num), dtype=torch.float)

        for i, d in enumerate(label["annotation"]["object"]):
            classes_name = d["name"]
            bbox_info = d["bndbox"]
            (bbox_x1, bbox_x2, bbox_y1, bbox_y2) = (float(bbox_info["xmin"]), float(bbox_info["xmax"]), float(bbox_info["ymin"]), float(bbox_info["ymax"]))
            bbox_x = (bbox_x1 + bbox_x2) / 2
            bbox_y = (bbox_y1 + bbox_y2) / 2
            bbox_h = bbox_y2 - bbox_y1
            bbox_w = bbox_x2 - bbox_x1

            bbox_x = bbox_x / img_w
            bbox_y = bbox_y / img_h
            bbox_w = bbox_w / img_w
            bbox_h = bbox_h / img_h

            bbox[i] = torch.tensor([bbox_x, bbox_y, bbox_h, bbox_w], dtype=torch.float)
            classes[i] = self.voc_dataset_prepare.label_name2onehot(classes_name)
        
        return (img_size, bbox, classes)
    
    def encode_yolo3_output(self, yolo3_output, normlize=True):
        encoded_list = []
        (C, _, _) = yolo3_output[0].shape
        assert C % self.anchor_num == 0
        split_num = C // self.anchor_num

        anchor_index = 0
        for feature_map in yolo3_output:
            (C, H, W) = feature_map.shape
            dev = feature_map.device
            tmp = torch.zeros(C, H, W).to(dev)
            x_bias_index = torch.arange(0, 1, 1.0/W).view(1, -1).to(dev)
            y_bias_index = torch.arange(0, 1, 1.0/H).view(-1, 1).to(dev)
            for i in range(self.anchor_num):
                (anchor_h, anchor_w) = self.anchor_list[anchor_index]
                start_ptr = i * split_num
                (x_center, y_center) = feature_map[start_ptr + 0:start_ptr + 2, :, :]
                (w, h) = feature_map[start_ptr + 2:start_ptr + 4, :, :]
                has_obj = feature_map[start_ptr + 4, :, :]
                classes = feature_map[start_ptr + 5:start_ptr + 5 + self.voc_dataset_prepare.classes_num, :, :]
                if normlize:
                    x_center = x_center.sigmoid()
                    y_center = y_center.sigmoid()
                    w = w.sigmoid()
                    h = h.sigmoid()
                    has_obj = has_obj.sigmoid()
                    classes = classes.softmax(dim=0)

                x_center = x_center + x_bias_index
                y_center = y_center + y_bias_index
                w = w * anchor_w
                h = h * anchor_h
                anchor_index += 1

                tmp[start_ptr + 0, :, :] = x_center
                tmp[start_ptr + 1, :, :] = y_center
                tmp[start_ptr + 2, :, :] = w
                tmp[start_ptr + 3, :, :] = h
                tmp[start_ptr + 4, :, :] = has_obj
                tmp[start_ptr + 5:start_ptr + 5 + self.voc_dataset_prepare.classes_num, :, :] = classes

            encoded_list.append(tmp)
        return encoded_list
    
    def label2tensor(self, label, feature_map_size, feature_map_level):
        (feature_h, feature_w) = feature_map_size
        anchor_num = self.anchor_num
        classes_num = self.voc_dataset_prepare.classes_num
        (img_size, bbox_list, classes_list) = self.decode_label(label)
        anchor_dim = 5 + classes_num
        output = torch.zeros((anchor_num * anchor_dim, feature_h, feature_w), dtype=torch.float)

        for (bbox, classes) in zip(bbox_list, classes_list):
            max_iou_arg = self.get_max_iou_arg(bbox, self.anchor_list)
            if max_iou_arg >= feature_map_level * anchor_num and max_iou_arg < (feature_map_level + 1) * anchor_num:
                # It means that this bbox belongs to the feature map level
                anchor = self.anchor_list[max_iou_arg]

                # Get index
                [bbox_x, bbox_y, bbox_h, bbox_w] = bbox
                [anchor_h, anchor_w] = anchor

                feature_map_index_h_float = bbox_y * feature_h
                feature_map_index_w_float = bbox_x * feature_w

                feature_map_index_h = int(feature_map_index_h_float)
                feature_map_index_w = int(feature_map_index_w_float)
                anchor_index_start = (max_iou_arg % anchor_num) * anchor_dim

                output[anchor_index_start + 0, feature_map_index_h, feature_map_index_w] = feature_map_index_w_float - feature_map_index_w
                output[anchor_index_start + 1, feature_map_index_h, feature_map_index_w] = feature_map_index_h_float - feature_map_index_h
                output[anchor_index_start + 2, feature_map_index_h, feature_map_index_w] = bbox_w
                output[anchor_index_start + 3, feature_map_index_h, feature_map_index_w] = bbox_h
                output[anchor_index_start + 4, feature_map_index_h, feature_map_index_w] = 1
                output[anchor_index_start + 5 : anchor_index_start + 5 + classes_num, feature_map_index_h, feature_map_index_w] = classes

        return output

def collate_fn(batch):
    x = torch.stack([i[0] for i in batch], dim=0)
    y = [i[1] for i in batch]
    return (x, y)

