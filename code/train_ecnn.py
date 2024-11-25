import traceback
from os import path

from ecnn.training import train_model
from ecnn.testing import test_model
from ecnn.network import NetworkBasic, NetworkAdvanced

def main():
    # Get the directory of the current file
    dir_ = path.dirname(path.abspath(__file__))
    print(f"Current directory: {dir_}")

    model = NetworkAdvanced
    model_name = "tcnn_advanced_5000"

    # Make sure you have already downloaded the data mentioned in the README file.
    train_params = {
        'model': model,
        'model_name': model_name,
        'images_folder': "train_color",
        'dir': dir_,
        'amount_images': 5000,
        'learning_rate': 0.01,
        'pin_memory': True,
        'prefetch_factor': 4,
        'num_workers': 4,
    }
    try:
        train_model(**train_params)

        # Puts to the test the final model.
        test_model(model, model_name, "test_color", dir_)
        
    except KeyboardInterrupt as e:
        print(e)
    except Exception as e:
        print(e)
        print(traceback.format_exc())


if __name__ == "__main__":
    main()
