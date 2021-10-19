from torchvision.datasets import CIFAR10, MNIST
from torchvision import transforms
import torch

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