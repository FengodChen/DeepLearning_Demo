import torch

class YOLO3_Loss(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, y_pred, y_true):
        pass