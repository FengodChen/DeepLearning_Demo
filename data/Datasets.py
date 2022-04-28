from torchvision.datasets import CIFAR10, MNIST, voc
from torchvision import transforms
from torch.utils.data.dataset import Dataset
import torch

class Imagenet_1k(Dataset):
    def __init__(self) -> None:
        super().__init__()
    
    def __len__(self):
        pass

    def __getitem__(self, index):
        pass

def cifar_dataset(root_dir, download=False, train=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.GaussianBlur(3),
        #transforms.Normalize((0, 0, 0), (1, 1, 1))
    ])
    ds = CIFAR10(root=root_dir, download=download, transform=transform, train=train)
    return ds

def mnist_dataset(root_dir, download=False):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.GaussianBlur(3),
        #transforms.Normalize((0, 0, 0), (1, 1, 1))
    ])
    ds = MNIST(root=root_dir, download=download, transform=transform)
    return ds

def voc_dataset(root_dir, year, resize, image_set="train", download=False):
    assert image_set in ["train", "val", "test"]
    transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor(),
        transforms.GaussianBlur(3),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    ds = voc.VOCDetection(root=root_dir, year=year, image_set=image_set, download=download, transform=transform)
    return ds
