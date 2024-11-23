from cnn.training import train_model
from cnn.testing import test_model, render_layer_output

# Make sure you have already downloaded the data mentioned in the README file.
train_model("./data/train_color", size=4)

# Puts to the test the final model.
test_model("./models/cnn_6th_2000_full.pt", "cnn_testing", "./data/test_color", architecture=6)

# Renders the outputs of the layers.
render_layer_output("./models/cnn_6th_2000_full.pt", 6, "", "./data/train_color", 16)