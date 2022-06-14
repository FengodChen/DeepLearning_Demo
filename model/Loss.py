import torch
from torch import nn

class YOLO3_Loss(nn.Module):
    def __init__(self, dataset_utils, lambda_coord, lambda_noobj) -> None:
        super().__init__()
        self.dataset_utils = dataset_utils
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.bce_loss = nn.BCELoss()
    
    def forward(self, y_pred, y_true):
        loss_sum = 0.0
        batch_size = y_pred[0].shape[0]

        for batch_i in range(batch_size):
            yolo3_pred = [b[batch_i] for b in y_pred]
            dataset_true = []
            for (feature_level, feature_map) in enumerate(yolo3_pred):
                (C, H, W) = feature_map.shape
                feature_map_size = (H, W)
                dataset_true.append(self.dataset_utils.label2tensor(y_true[batch_i], feature_map_size, feature_level))

            yolo3_pred = self.dataset_utils.encode_yolo3_output(yolo3_pred)
            dataset_true = self.dataset_utils.encode_yolo3_output(dataset_true, normlize=False)

            for (feature_map_true, feature_map_pred) in zip(dataset_true, yolo3_pred):
                loss_sum += self.get_feature_map_loss(feature_map_true, feature_map_pred)
        
        return loss_sum / (batch_i+1)
    
    def get_feature_map_loss(self, feature_map_true, feature_map_pred):
        anchor_num = self.dataset_utils.anchor_num
        anchor_dim =  5 + self.dataset_utils.classes_num

        loss = 0.0
        for i in range(anchor_num):
            start_ptr = i * anchor_dim
            end_ptr = start_ptr + anchor_dim
            split_feature_map_true = feature_map_true[start_ptr:end_ptr, ...]
            split_feature_map_pred = feature_map_pred[start_ptr:end_ptr, ...]
            loss += self.get_splited_feature_map_loss(split_feature_map_true, split_feature_map_pred)
        return loss
    
    def get_splited_feature_map_loss(self, feature_map_true, feature_map_pred):
        loss = 0.0
        has_obj_ptr = torch.where(feature_map_true[4, :, :] >= 0.9)
        no_obj_ptr = torch.where(feature_map_true[4, :, :] <= 0.1)

        if (len(has_obj_ptr[0]) > 0):
            (x_p, y_p, h_p, w_p) = (feature_map_pred[i][has_obj_ptr] for i in range(4))
            (x_t, y_t, h_t, w_t) = (feature_map_true[i][has_obj_ptr] for i in range(4))
            obj_p = feature_map_pred[4][has_obj_ptr]
            obj_t = feature_map_true[4][has_obj_ptr]
            classes_p = feature_map_pred[5:].permute(1, 2, 0)[has_obj_ptr]
            classes_t = feature_map_true[5:].permute(1, 2, 0)[has_obj_ptr]

            loss += self.lambda_coord * ((x_p - x_t)**2 + (y_p - y_t)**2).sum()
            loss += self.lambda_coord * ((h_p**0.5 - h_t**0.5)**2 + (w_p**0.5 - w_t**0.5)**2).sum()
            loss += self.bce_loss(obj_p, obj_t)
            loss += self.bce_loss(classes_p, classes_t)

        if (len(no_obj_ptr[0]) > 0):
            noobj_p = feature_map_pred[4][no_obj_ptr]
            noobj_t = feature_map_true[4][no_obj_ptr]
            loss += self.lambda_noobj * self.bce_loss(noobj_p, noobj_t)
        
        return loss


