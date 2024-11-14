from tcnn.training import train_model
from tcnn.testing import test_model
from tcnn.network import NetworkBasic, NetworkAdvanced
from os import path


def main():
    # Get the directory of the current file
    dir_ = path.dirname(path.abspath(__file__))
    print(f"Current directory: {dir_}")

    model = NetworkAdvanced
    model_name = "tcnn_advanced_1024"

    # Make sure you have already downloaded the data mentioned in the README file.
    train_params = {
        'model': model,
        'model_name': model_name,
        'images_folder': "train_color",
        'dir': dir_,
        'amount_images': 1024,
        'learning_rate': 0.01,
        'pin_memory': True,
        'prefetch_factor': 4,
        'num_workers': 4,
    }
    train_model(**train_params)

    # Puts to the test the final model.
    test_model(model, model_name, "test_color", dir_)


if __name__ == "__main__":
    main()
