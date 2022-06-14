import torch

def cifar10_compare_func(y_pred, y):
    y_pred_ = torch.argmax(y_pred, dim=1).view(-1)
    y_ = y.view(-1)
    true_ans = y_-y_pred_
    true_ans = true_ans[true_ans == 0]
    acc = len(true_ans) / len(y_)
    return acc

def void_compare_func(y_pred, y):
    return 0