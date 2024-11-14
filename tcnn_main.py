from tcnn.training import train_model
from tcnn.testing import test_model
from tcnn.network import NetworkBasic, NetworkAdvanced
from os import path

def main():
    # Get the directory of the current file
    dir = path.dirname(path.abspath(__file__))
    print(f"Current directory: {dir}")

    model = NetworkAdvanced
    model_name = "tcnn_basic"
    
    # Make sure you have already downloaded the data mentioned in the README file.
    train_model(model, model_name, "train_color", dir, amount_images=64)

    # Puts to the test the final model.
    test_model(model, model_name, "test_color", dir)
    
if __name__ == "__main__":
    main()
