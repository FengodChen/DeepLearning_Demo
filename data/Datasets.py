from torchvision.datasets import CIFAR10, MNIST, voc, CocoDetection
from torchvision import transforms
from torch.utils.data.dataset import Dataset
import torch

class RandomDataset1(Dataset):
    def __init__(self, x_dim, x_num, y_dim, len=1024):
        self.x_dim = x_dim
        self.x_num = x_num
        self.y_dim = y_dim

        self.len = len

    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        x = torch.rand(self.x_dim, self.x_num)
        y = torch.sum(x, dim=0) / self.x_dim
        return (x, y)

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
    def __init__(self, root_dir, year, resize, image_set="train", download=False, for_show=False) -> None:
        super().__init__()
        assert image_set in ["train", "val", "test"]
        if for_show:
            transform = transforms.Compose([
                transforms.Resize(resize),
                transforms.ToTensor(),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(resize),
                transforms.ToTensor(),
                transforms.GaussianBlur(3),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        self.ds = voc.VOCDetection(root=root_dir, year=year, image_set=image_set, download=download, transform=transform)
        self.voc_util = None
    
    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        x = self.ds[index][0]
        y = self.ds[index][1]
        if self.voc_util is not None:
            y = self.voc_util.label2tensor(y)
            y = self.voc_util.encode_to_tensor(y, "label")
            return [x] + [i for i in y]
        else:
            return [x, y]
    
    def set_voc_util(self, voc_util):
        self.voc_util = voc_util
    
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