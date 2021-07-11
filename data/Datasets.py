from torchvision.datasets import CIFAR10
from torchvision import transforms
import torch

def cifar_dataset(root_dir, download=False):
	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.GaussianBlur(3),
		#transforms.Normalize((0, 0, 0), (1, 1, 1))
	])
	ds = CIFAR10(root=root_dir, download=download, transform=transform)
	return ds