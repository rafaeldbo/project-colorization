from os import path, mkdir
from tqdm import tqdm

import torch
from torch import nn, no_grad, load
from torch.utils.data import DataLoader

from .dataset import ImageDataset
from .utilities import gen_compare_figure


def test_model(
    model: nn.Module,
    model_name: str,
    images_folder: str,
    dir: str = "./",
    images_amount=-1,
    batch_size=32,
    cuda=True,
    gen_images=False,
):
    # Loading Dataset and creating the corresponding DataLoader.
    dataset = ImageDataset(dir, images_folder, images_amount)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    output_dir = path.join(dir, "output")
    if not path.exists(output_dir):
        mkdir(output_dir)

    model_dir = path.join(dir, "models", f"{model_name}.pt")
    if not path.isfile(model_dir):
        raise Exception(
            "No model found. Please train the model first or make sure the model name is correct."
        )
        
    # choosing the device
    device_name = "cuda" if torch.cuda.is_available() and cuda else "cpu"
    print(f"Testing using [{device_name}]")
    device = torch.device(device_name)

    # Loading the NN and passing the data.
    ecnn = model().to(device)
    ecnn.load_state_dict(load(model_dir, weights_only=True)["model_state_dict"])

    total_loss = 0  
    
    print("Testing the model...")  
    total_batches = len(dataloader)
    progress_bar = tqdm(total=total_batches, desc="Testing Progress", position=0, bar_format='{l_bar}{bar:20}{r_bar}')
    # Creating an image grid to showcase the results.
    for i, batch in enumerate(dataloader):
        gray, color, category = batch

        gray = gray.to(device)
        color = color.to(device)
        category = category.to(device)
        
        predicted = ecnn(gray, category).int()
        
        with no_grad():
            criterion = nn.MSELoss()
            loss = criterion(predicted, color)
            
            total_loss += loss.item()

        if gen_images:
            filename = path.join(output_dir, f"output_{model_name}_{i}.png")
            gen_compare_figure(gray, color, predicted, filename)
        
        # Updating the progress bar.
        progress_bar.update(1)
        progress_bar.set_postfix(batch=i+1, loss=loss.item())
    progress_bar.close()
            
    print("Testing finished!")
    print(f"Mean Loss in Test Dataset: {round(total_loss/total_batches, 2)}")