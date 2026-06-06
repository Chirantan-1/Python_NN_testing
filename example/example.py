import numpy as np
from tensorflow.keras.datasets import mnist
from TheNN import NeuralNetwork

(X, Y), (x, y) = mnist.load_data()

X = X.reshape(-1, 784) / 255.0
x = x.reshape(-1, 784) / 255.0
Y = np.eye(10)[Y]

nn = NeuralNetwork(save_file="model.pkl")

nn.add_layer(784, 128, "relu")
nn.add_layer(128, 64, "tanh")
nn.add_layer(64, 32, "tanh")
nn.add_layer(32, 10, "softmax")

nn.train(X, Y, x, y, batch=200, epochs=50, lr=0.05)
