import pandas as pd

from os import path, listdir
from torch import from_numpy, Tensor
from torch.utils.data import Dataset

from skimage.color import rgb2lab
from skimage.io import imread


class ImageDataset(Dataset):
    """
    A dataset that contains the colored images and their corresponding categories.
    """
    
    def __init__(self, dir_path:str, images_folder:str, size:int=-1) -> None:
        self.data_dir = path.join(dir_path, 'data')
        self.images_path = path.join(self.data_dir, images_folder)
        
        images = listdir(self.images_path) 
        size = size if size > 0 else len(images)
        self.images_files = images[:size]
        
        df = pd.read_csv(path.join(self.data_dir, 'categories.csv'), delimiter=';')
        df['category'] = df['category'].fillna(0)
        self.categories = df.set_index('image')['category'].to_dict()

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
        LAB_image = from_numpy(rgb2lab(imread(image_path))).permute(2, 0, 1)

        # The gray image consists of the `L` part of the LAB format.
        gray_layer = LAB_image[0, :, :].unsqueeze(0)
        # The color image consists of the 'a' and 'b' parts of the LAB format.
        color_layers = LAB_image[1:, :, :]

        return gray_layer.float(), color_layers.float(), self.categories[filename]
