import matplotlib.pyplot as plt

from torch import Tensor, nn, no_grad, cat, from_numpy
from skimage.color import lab2rgb

def plot_loss(loss, title, filename):
    """
    Plots the loss for each epoch.
    :param loss: A list with the loss per epoch.
    :param title: The plot title.
    :param filename: The filename.
    """
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(loss)
    plt.savefig(f"figures/{filename}.png")


def plot_losses(losses, labels, title, filename):
    """
    Plots multiple losses for each epoch.
    :param losses: A 2D list with the losses per epoch.
    :param labels: The label for each case.
    :param title: The plot title.
    :param filename: The filename.
    """
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    for i in range(len(losses)):
        plt.plot(losses[i], label=labels[i])
    plt.legend()
    plt.savefig(f"figures/{filename}.png")

def parse_str(string:str) -> str|int|float:
    if string.isdigit():
        return int(string)
    if string.replace('.', '', 1).isdigit():
        return float(string)
    return string

def read_csv(file:str, delimiter:str=',') -> dict|list:
    
    with open(file) as f:
        first_line = f.readline()
        if delimiter not in first_line:
            raise ValueError('Invalid delimiter')
        first_line = first_line.strip().split(delimiter)
        columns = len([column for column in first_line if (column not in ['', ' ', '\n', None])])
        data = {}
        if columns == 0:
            raise ValueError('No columns found')
        if columns == 1:
            return [parse_str(line.strip()) for line in f]
        if columns == 2:
            return {key: parse_str(value) for key, value in (line.strip().split(delimiter) for line in f)}
        else:
            return {key: [*value] for key, *value in (line.strip().split(delimiter) for line in f)}
    return pd.read_csv(file)

def gen_compare_figure(gray:Tensor, color:Tensor, predicted:Tensor, filename:str, columns:int) -> None:
    
    gray_images = gray.squeeze() # Removing the channel.
    
    fig = plt.figure(figsize=(8, 8))

    # First row consists of the gray images.
    for i in range(1, columns + 1):
        fig.add_subplot(3, columns, i)
        img = gray_images[i - 1]

        # Setting the title on top of the middle image.
        if i == 3:
            plt.title("Input")

        plt.gca().get_xaxis().set_visible(False)
        plt.gca().get_yaxis().set_visible(False)
        plt.imshow(img, cmap="gray")

    # Second row consists of the actual colored images.
    for i in range(1, columns + 1):
        fig.add_subplot(3, columns, i + columns)
        # Setting the title on top of the middle image.
        if i == 3:
            plt.title("Actual")

        img = cat((gray[i - 1], color[i - 1]), 0)
        img = lab2rgb(img.permute(1, 2, 0))
        plt.gca().get_xaxis().set_visible(False)
        plt.gca().get_yaxis().set_visible(False)
        plt.imshow(img)

    # Third row consists of the predicted colored images.
    with no_grad():
        for i in range(1, columns + 1):
            fig.add_subplot(3, columns, i + 2 * columns)
            mse = nn.MSELoss()

            img = cat((gray[i - 1], predicted[i - 1]), 0)
            img = lab2rgb(img.permute(1, 2, 0))
            img_true = cat((gray[i - 1], color[i - 1]), 0)
            img_true = lab2rgb(img_true.permute(1, 2, 0))
            loss = mse(from_numpy(img).permute(2, 0, 1), from_numpy(img_true).permute(2, 0, 1))
            plt.gca().get_yaxis().set_visible(False)
            plt.gca().set_xticks([])
            plt.gca().set_xlabel(f"{loss.item():.5f}")

            # Setting the title on top of the middle image.
            if i == 3:
                plt.title("Predicted")
            
            plt.imshow(img)

    plt.savefig(filename)