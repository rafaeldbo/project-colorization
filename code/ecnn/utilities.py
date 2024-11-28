import matplotlib.pyplot as plt

import torch
from torch import Tensor, no_grad, cat
from skimage.color import lab2rgb

import warnings 
warnings.simplefilter("ignore")

def gen_compare_figure(gray:Tensor, color:Tensor, predicted:Tensor, filename:str) -> None:
    
    gray = gray.cpu()
    color = color.cpu()
    predicted = predicted.cpu()
    
    gray_images = gray.squeeze() # Removing the channel.
    
    fig = plt.figure(figsize=(8, 8))
    columns = 5 if len(gray) >= 5 else len(gray)
    middle = columns//2

    with no_grad():
        # First row consists of the gray images.
        for i in range(columns):
            fig.add_subplot(3, columns, i+1)
            img = gray_images[i]
            # Setting the title on top of the middle image.
            if i == middle: plt.title("Input")
            plt.imshow(img, cmap="gray")
            plt.axis("off")

        real_img = lab2rgb(cat((gray, color), 1).permute(0, 2, 3, 1))
        # Second row consists of the actual colored images.
        for i in range(columns):
            fig.add_subplot(3, columns, i+columns+1)
            # Setting the title on top of the middle image.
            if i == middle:
                plt.title("Actual")
            plt.imshow(real_img[i])
            plt.axis("off")

        predicted_img = lab2rgb(cat((gray, predicted), 1).permute(0, 2, 3, 1))
        # Third row consists of the predicted colored images.
        
        for i in range(columns):
            fig.add_subplot(3, columns, i+(2*columns)+1)
            # Setting the title on top of the middle image.
            if i == middle: plt.title("Predicted")
            plt.imshow(predicted_img[i])
            plt.axis("off")

    plt.savefig(filename)