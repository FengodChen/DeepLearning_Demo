from torch.utils.data.dataloader import DataLoader
from data.Datasets import voc_dataset

class K_Mean:
    def __init__(self, root_dir, year, resize) -> None:
        self.dataset = voc_dataset(root_dir=root_dir, year=year, resize=resize, image_set="train")