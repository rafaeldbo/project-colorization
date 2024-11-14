from os import path, mkdir
from torch import nn, load
from torch.utils.data import DataLoader

from .dataset import ImageDataset
from .utilities import gen_compare_figure


def test_model(model:nn.Module, model_name:str, images_folder:str, dir:str='./', show_images=True):
    # Loading Dataset and creating the corresponding DataLoader.
    data = ImageDataset(dir, images_folder)
    data_loader = DataLoader(data, batch_size=4, shuffle=False)

    output_dir = path.join(dir, "output")
    if not path.exists(output_dir):
        mkdir(output_dir)
        
    model_dir = path.join(dir, "models", f"{model_name}.pt")
    if not path.isfile(model_dir):
        raise Exception("No model found. Please train the model first or make sure the model name is correct.")
            
    # Loading the NN and passing the data.
    cnn = model()
    cnn.load_state_dict(load(model_dir, weights_only=True)["model_state_dict"])

    # Grid setup.
    columns = 4

    print("Testing the model.")
    # Creating an image grid to showcase the results.
    for j,  batch in enumerate(data_loader):
        gray, color, category = batch
        if gray is None:
            break
        
        gray = gray
    
        predicted = cnn(gray, category, category).int()

        if show_images:
            filename = path.join(output_dir, f"output_{model_name}_{j}.png")
            gen_compare_figure(gray, color, predicted, filename, columns) 
        