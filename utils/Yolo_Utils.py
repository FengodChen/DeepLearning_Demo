import pickle
from torch.utils.data.dataloader import DataLoader
import torchvision
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
    
    def label_onehot2name(self, onehot):
        onehot = onehot.view(-1)
        num = int(torch.argmax(onehot))
        return self.label_name[num]
    
    def load(self, file_path):
        with open(file_path, "rb") as f:
            self.label_name = pickle.load(f)
    
    def save(self, file_path):
        with open(file_path, "wb") as f:
            pickle.dump(self.label_name, f)

class VOC_Utils:
    def __init__(self, img_size, voc_kmeans:Voc_Kmeans, voc_dataset_prepare:Voc_Dataset_Prepare, net, anchor_num:int, dev, img_channel=3):
        assert len(voc_kmeans.cluster_list) % anchor_num == 0

        self.voc_kmeans = voc_kmeans
        self.voc_dataset_prepare = voc_dataset_prepare

        (self.img_h, self.img_w) = self.img_size = img_size

        self.voc_kmeans.sort()
        self.anchor_list = self.voc_kmeans.cluster_list
        self.anchor_num = anchor_num
        self.classes_num = self.voc_dataset_prepare.classes_num
        self.img_channel = img_channel
        self.net = net
        self.feature_map_size_list = []
        self.dev = dev

        self.__init_feature_map_list__()
    
    def __init_feature_map_list__(self):
        dev = self.dev
        x = torch.rand((1, self.img_channel, self.img_h, self.img_w), device=dev)
        feature_map_list = self.net(x)
        for feature_map in feature_map_list:
            (B, C, H, W) = feature_map.shape
            self.feature_map_size_list.append((C, H, W))
        
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
    
    def encode_to_tensor(self, yolo3_output, encode_type, normalize=False):
        assert encode_type in ["label", "yolo3_output"]
        (C, _, _) = yolo3_output[0].shape
        assert C % self.anchor_num == 0
        split_num = C // self.anchor_num

        feature_level_len = len(yolo3_output)
        anchor_index = 0
        for feature_level in range(feature_level_len):
            feature_map = yolo3_output[feature_level]
            (C, H, W) = feature_map.shape
            dev = feature_map.device
            x_bias_index = torch.arange(0, 1, 1.0/W).view(1, -1).to(dev)
            y_bias_index = torch.arange(0, 1, 1.0/H).view(-1, 1).to(dev)
            for i in range(self.anchor_num):
                (anchor_h, anchor_w) = self.anchor_list[anchor_index]
                start_ptr = i * split_num
                (x_center, y_center) = feature_map[start_ptr + 0:start_ptr + 2, :, :]
                (w, h) = feature_map[start_ptr + 2:start_ptr + 4, :, :]
                has_obj = feature_map[start_ptr + 4, :, :]
                classes = feature_map[start_ptr + 5:start_ptr + 5 + self.voc_dataset_prepare.classes_num, :, :]
                if encode_type == "yolo3_output":
                    x_center = x_center.sigmoid()
                    y_center = y_center.sigmoid()
                    w = w
                    h = h
                    has_obj = has_obj.sigmoid()
                    classes = classes.softmax(dim=0)

                x_center = (x_center / W + x_bias_index) * self.img_w
                y_center = (y_center / H + y_bias_index) * self.img_h
                if encode_type == "label":
                    w = w * self.img_w
                    h = h * self.img_h
                elif encode_type == "yolo3_output":
                    w = torch.exp(w) * anchor_w * self.img_w
                    h = torch.exp(h) * anchor_h * self.img_h
                anchor_index += 1

                if normalize:
                    x_center = x_center / self.img_w
                    y_center = y_center / self.img_h
                    w = w / self.img_w
                    h = h / self.img_h

                feature_map[start_ptr + 0, :, :] = x_center
                feature_map[start_ptr + 1, :, :] = y_center
                feature_map[start_ptr + 2, :, :] = w
                feature_map[start_ptr + 3, :, :] = h
                feature_map[start_ptr + 4, :, :] = has_obj
                feature_map[start_ptr + 5:start_ptr + 5 + self.voc_dataset_prepare.classes_num, :, :] = classes

        return yolo3_output
    
    def yolo3_encode_to_tensor(self, y):
        batch_size = y[0].shape[0]
        for batch_i in range(batch_size):
            l = []
            for batch_feature_i in range(len(y)):
                batch_feature_map = y[batch_feature_i]
                l.append(batch_feature_map[batch_i])
            ll = self.encode_to_tensor(l, "yolo3_output", normalize=True)
            for i in range(len(l)):
                l[i][:] = ll[i][:]
        return y
    
    def label2tensor_per_level(self, label, feature_map_size, feature_map_level):
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
    
    def label2tensor(self, label):
        y = []
        for (feature_map_level, feature_map_shape) in enumerate(self.feature_map_size_list):
            (C, H, W) = feature_map_shape
            y.append(self.label2tensor_per_level(label, (H, W), feature_map_level))
        return y

    
    def decode_tensor(self, yolo3_output, decode_type, has_obj_thread):
        assert self.anchor_num % len(yolo3_output) == 0

        bbox_list = []
        label_list = []

        encoded_yolo3_output = self.encode_to_tensor(yolo3_output, encode_type=decode_type)
        dim_per_anchor = 5 + self.classes_num
        for feature_map_i in range(len(encoded_yolo3_output)):
            feature_map = encoded_yolo3_output[feature_map_i]
            for anchor_i in range(self.anchor_num):
                start_ptr = anchor_i * dim_per_anchor
                end_ptr = start_ptr + dim_per_anchor
                map_ptr = torch.where(feature_map[start_ptr+4, :, :] > has_obj_thread)
                anchors = feature_map.permute(1, 2, 0)[map_ptr][:, start_ptr:end_ptr]
                for anchor_i in range(len(anchors)):
                    anchor = anchors[anchor_i]
                    (x_center, y_center, w, h) = anchor[0:4]
                    classes = anchor[5:]

                    w = float(w)
                    h = float(h)
                    x_center = float(x_center)
                    y_center = float(y_center)

                    x_min = int(x_center - w//2)
                    x_max = int(x_center + w//2)
                    y_min = int(y_center - h//2)
                    y_max = int(y_center + h//2)

                    classes = self.voc_dataset_prepare.label_onehot2name(classes)

                    bbox_list.append([x_min, y_min, x_max, y_max])
                    label_list.append(classes)
        
        return (bbox_list, label_list)

    
    def draw_bbox(self, img, yolo3_predict, decode_type, has_obj_thread):
        yolo3_output = []
        for b in yolo3_predict:
            (B, _, _, _) = b.shape
            assert B == 1
            yolo3_output.append(b[0])
        (bbox_list, label_list) = self.decode_tensor(yolo3_output, decode_type, has_obj_thread)
        bbox_list = torch.tensor(bbox_list)
        drawed_img = torchvision.utils.draw_bounding_boxes(img, bbox_list, label_list, fill=True)
        return drawed_img


def collate_fn(batch):
    x = torch.stack([i[0] for i in batch], dim=0)
    y = [i[1] for i in batch]
    return (x, y)

