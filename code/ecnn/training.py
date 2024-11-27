import time
from os import mkdir, path
from tqdm import tqdm 

import torch
from torch import load, nn, optim, save
from torch.utils.data import DataLoader

from .dataset import ImageDataset


def train_model(
    model: nn.Module,
    model_name: str,
    images_folder: str,
    dir: str = "./",
    from_checkpoint: bool = False,
    checkpoints_amount: int = 1,
    amount_images: int = 4,
    epochs: int = 50,
    learning_rate: float = 0.001,
    batch_size: int = 32,
    cuda: bool = True,
    pin_memory: bool = False,
    prefetch_factor: int = None,
    num_workers: int = 0,
) -> None:

    # Checking multiprocessing parameters are valid.
    if prefetch_factor is not None and num_workers == 0:
        raise Exception("You need to have at least one worker to use the prefetch_factor.")

    # Loading Dataset and creating the corresponding DataLoader.
    print("Loading dataset.")
    batch = ImageDataset(dir, images_folder, amount_images)
    dataloader = DataLoader(
        batch,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        num_workers=num_workers,
    )

    # Creating folders to store the checkpoints and models.
    checkpoints_dir = path.join(dir, "checkpoints")
    if not path.exists(checkpoints_dir):
        mkdir(checkpoints_dir)
        from_checkpoint = False
    checkpoint_file = path.join(checkpoints_dir, f"{model_name}_checkpoint.pt")
    checkpoint_period = epochs // (checkpoints_amount+1)

    models_dir = path.join(dir, "models")
    if not path.exists(models_dir):
        mkdir(models_dir)

    # choosing the device
    device_name = "cuda" if torch.cuda.is_available() and cuda else "cpu"
    print(f"Training using [{device_name}]")
    device = torch.device(device_name)

    ecnn = model().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(ecnn.parameters(), lr=learning_rate)

    running_losses = []

    # Checking if there is a pre-trained model to be loaded.
    if from_checkpoint:
        if path.isfile(checkpoint_file):
            print("Loading checkpoint")
            checkpoint = load(checkpoint_file, weights_only=True)
            ecnn.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            ecnn.eval()
            ecnn.train()
            initial_time = checkpoint["time"]
            running_losses = checkpoint["running_losses"]
        else:
            print("No checkpoint found. Training from scratch.")
        initial_time = 0
    else:
        initial_time = 0

    print(f"Number of parameters: {sum(p.numel() for p in ecnn.parameters())}")
    
    total_batches = epochs * len(dataloader)
    progress_bar = tqdm(total=total_batches, desc="Training Progress", position=0, bar_format='{l_bar}{bar:20}{r_bar}')

    try:
        start = time.time()
        for epoch in range(epochs):
            epoch_running_loss = 0
            for i, batch in enumerate(dataloader):
                gray, color, category = batch

                gray = gray.to(device)
                color = color.to(device)
                category = category.to(device)

                optimizer.zero_grad()
                outputs = ecnn(gray, category)

                loss = criterion(outputs, color)
                loss.backward()
                optimizer.step()

                epoch_running_loss += loss.item()
                
                # Updating the progress bar.
                progress_bar.update(1)
                progress_bar.set_postfix(epoch=epoch+1, batch=i+1, loss=loss.item())
            running_losses.append(epoch_running_loss)            

            # Saving the model every checkpoint_period epochs.
            current_epoch = epoch + 1
            if ((current_epoch) % checkpoint_period == 0) and (current_epoch < epochs):
                print("\nSaving checkpoint...")
                save(
                    {
                        "epoch": epoch,
                        "model_state_dict": ecnn.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "time": time.time() - start + initial_time,
                        "running_losses": running_losses,
                    },
                    f"./checkpoints/{model_name}_checkpoint.pt",
                )

        progress_bar.close()
        print(f"Training finished!")
        # Saving the final model. 
        save(
            {
                "epoch": epoch,
                "model_state_dict": ecnn.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "time": time.time() - start + initial_time,
                "running_losses": running_losses
            }, f"./models/{model_name}.pt",
        )
        
    except KeyboardInterrupt:
        progress_bar.close()
        print("\nSaving checkpoint...")
        save(
            {
                "epoch": epoch,
                "model_state_dict": ecnn.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "time": time.time() - start + initial_time,
                "running_losses": running_losses
            }, f"./checkpoints/{model_name}_checkpoint.pt",
        )
        raise KeyboardInterrupt("Training interrupted!")
