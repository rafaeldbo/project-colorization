import pandas as pd

from os import path, listdir
from torch import from_numpy, Tensor
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
from skimage.color import rgb2lab

from .utilities import read_csv


class ImageDataset(Dataset):
    """
    A dataset that contains the colored images and their corresponding categories.
    """
    
    def __init__(self, dir:str, images_folder:str, size:int=4) -> None:
        self.data_dir = path.join(dir, 'data')
        self.images_path = path.join(self.data_dir, images_folder)
        
        if size == 0:
            raise ValueError('Size cannot be zero')
        if size == -1:
            self.images_files = listdir(self.images_path)
        else:
            self.images_files = listdir(self.images_path)[:size]
        
        self.categories = read_csv(path.join(self.data_dir, 'categories.csv'), delimiter=';')

    def get_category(self, filename:str) -> int:
        """
        Fetches the category of the given filename.
        
        :param str filename: The filename of the image.
        :return int: The category of the image.
        """
        category = self.categories[filename.split('.')[0]]
        if category in ['', ' ', None]:
            return 0
        return category

    def __len__(self) -> int:
        """
        :return int: The length of the dataset.
        """
        return len(self.images_files)

    def __getitem__(self, index:int) -> tuple[Tensor, Tensor, int]:
        """
        Fetches the image data of the given index.
        
        :param int index: The index where the item is located at.
        :return tuple[tensor, tensor, int]: The color image, the gray image and the category in the given index.
        """
        
        filename = self.images_files[index]

        image_path = path.join(self.images_path, filename)
        image = from_numpy(rgb2lab(read_image(image_path).permute(1, 2, 0))).permute(2, 0, 1)

        # The color image consists of the 'a' and 'b' parts of the LAB format.
        color_image = image[1:, :, :]
        # The gray image consists of the `L` part of the LAB format.
        gray_image = image[0, :, :].unsqueeze(0)

        return gray_image.float(), color_image.float(), self.get_category(filename)
