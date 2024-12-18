import traceback
from os import path

from ecnn.training import train_model
from ecnn.testing import test_model
from ecnn.network import NetworkBasic, NetworkAdvanced

def main():
    # Get the directory of the current file
    dir_path = path.dirname(path.abspath(__file__))
    print(f"Current directory: {dir_path}")

    model = NetworkAdvanced
    model_name = "ecnn_advanced_5000"

    # Make sure you have already downloaded the data mentioned in the README file.
    train_params = {
        'model': model,
        'model_name': model_name,
        'images_folder': "train_color",
        'dir': dir_path,
        'amount_images': 5000, 
        'epochs': 100,
        'learning_rate': 0.001,
        'batch_size': 64,
        'pin_memory': True,
        'prefetch_factor': 4,
        'num_workers': 4,
    }
    try:
        # Training the model.
        train_model(**train_params)

        # Testing the model in the train dataset.
        test_model(model, model_name, "train_color", dir_path)

        # Testing the model in the test dataset.
        test_model(model, model_name, "test_color", dir_path)
        
    except KeyboardInterrupt as e:
        print(e)
    except Exception as e:
        print(e)
        print(traceback.format_exc())


if __name__ == "__main__":
    main()
