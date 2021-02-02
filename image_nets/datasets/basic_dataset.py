from torch.utils.data import Dataset
from typing import Callable, Optional

from image_nets.datareaders.base_datareader import DataReader
from image_nets.datareaders.get_datareaders import get_datareader


class BasicDataset(Dataset):

    def __init__(self, dataset_name: str, split: str, preprocessing_fn: Optional[Callable] = None,
                 augmentation: bool = False):

        self.datareader: DataReader = get_datareader(dataset_name)
        self.images = [images for images in self.datareader.get_split(split)]

        self.augmentation: bool = augmentation
        self.preprocessing_fn: Optional[Callable] = preprocessing_fn

    def __len__(self):
        return len(self.images)

    def __getitem__(self, ix: int):

        image, mask = self.images[ix]










