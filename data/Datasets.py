from torchvision.datasets import CIFAR10, MNIST, voc, CocoDetection
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

class Voc_Dataset(Dataset):
    def __init__(self, root_dir, year, resize, image_set="train", download=False) -> None:
        super().__init__()
        assert image_set in ["train", "val", "test"]
        transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor(),
            transforms.GaussianBlur(3),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.ds = voc.VOCDetection(root=root_dir, year=year, image_set=image_set, download=download, transform=transform)
    
    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        x = self.ds[index][0]
        y = self.ds[index][1]
        return (x, y)

class Coco_Dataset(Dataset):
    def __init__(self, img_dir, ann_path, resize) -> None:
        super().__init__()
        transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor(),
            transforms.GaussianBlur(3),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.ds = CocoDetection(img_dir, ann_path, transform=transforms)
    
    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        x = self.ds[index][0]
        y = self.ds[index][1]
        return (x, y)