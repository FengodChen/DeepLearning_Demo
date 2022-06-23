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
        y_true = self.dataset_utils.tensor_embedding(y_true)
        y_pred = self.dataset_utils.tensor_embedding(y_pred)

        return self.get_feature_map_loss(y_true, y_pred)

    def get_feature_map_loss(self, feature_map_true, feature_map_pred):
        # Shape = [(B H W), N, D]
        pos_mask = feature_map_true[..., 4] > 0.9
        ignore_mask = feature_map_true[..., 0] < -0.9
        neg_mask = ~(pos_mask | ignore_mask)

        # pos tensor
        pos_tensor_true = feature_map_true[pos_mask]
        pos_tensor_pred = feature_map_pred[pos_mask]
        (x_p, y_p, h_p, w_p) = (pos_tensor_pred[:, i] for i in range(4))
        (x_t, y_t, h_t, w_t) = (pos_tensor_true[:, i] for i in range(4))
        obj_p = pos_tensor_pred[:, 4]
        obj_t = pos_tensor_true[:, 4]
        classes_p = pos_tensor_pred[:, 5:]
        classes_t = pos_tensor_true[:, 5:]

        pos_loss = torch.zeros(1)
        pos_loss += self.lambda_coord * ((x_p - x_t)**2 + (y_p - y_t)**2).mean()
        pos_loss += self.lambda_coord * ((h_p**0.5 - h_t**0.5)**2 + (w_p**0.5 - w_t**0.5)**2).mean()
        pos_loss += self.bce_loss(obj_p, obj_t)
        pos_loss += self.bce_loss(classes_p, classes_t)

        # neg tensor
        neg_tensor_true = feature_map_true[neg_mask]
        neg_tensor_pred = feature_map_pred[neg_mask]
        noobj_p = neg_tensor_pred[:, 4]
        noobj_t = neg_tensor_true[:, 4]
        neg_loss = self.lambda_noobj * self.bce_loss(noobj_p, noobj_t)

        return pos_loss + neg_loss
    

