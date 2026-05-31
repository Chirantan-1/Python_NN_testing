from TheNN import NeuralNetwork
from NNDrawer import NNDrawer

nn = NeuralNetwork(save_file="model.pkl")

nn.load()

drawer = NNDrawer(
    nn=nn,
    input_size=784,
    screen_size=600,
    output_size=10,
    output_labels=["0","1","2","3","4","5","6","7","8","9"],
    rows=28,
    cols=28
)

drawer.run("Digit Recognition")
