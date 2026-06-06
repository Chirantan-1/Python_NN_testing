import numpy as np
import pickle

class NeuralNetwork:

    VA = {"relu", "sigmoid", "tanh", "softmax", "linear"}

    def __init__(self, save_file=None, loss="cross_entropy"):
        self.ls = []
        self.save_file = save_file
        self.l = loss
        self.classf = loss in {"cross_entropy", "binary_cross_entropy"}

    def add_layer(self, inp, out, a):

        if a not in self.VA:
            raise ValueError(f"invalid activation: {a}")

        if a == "relu":
            W = np.random.randn(inp, out) * np.sqrt(2 / inp)
        else:
            limit = np.sqrt(6 / (inp + out))
            W = np.random.uniform(-limit, limit, (inp, out))

        B = np.zeros((1, out))

        self.ls.append({"W": W, "B": B, "act": a})

    def act(self, Z, a):

        if a == "relu":
            return np.maximum(0, Z)

        if a == "sigmoid":
            return 1 / (1 + np.exp(-np.clip(Z, -500, 500)))

        if a == "tanh":
            return np.tanh(Z)

        if a == "softmax":
            exp = np.exp(Z - np.max(Z, axis=1, keepdims=True))
            return exp / np.sum(exp, axis=1, keepdims=True)

        if a == "linear":
            return Z

    def actd(self, Z, a):

        if a == "relu":
            return (Z > 0).astype(float)

        if a == "sigmoid":
            s = self.act(Z, "sigmoid")
            return s * (1 - s)

        if a == "tanh":
            return 1 - np.tanh(Z) ** 2

        if a == "linear":
            return np.ones_like(Z)

        raise ValueError("Softmax without cross_entropy")

    def loss(self, A, Y):

        eps = 1e-8

        if self.l == "mse":
            return np.mean((A - Y) ** 2) / 2

        if self.l == "mae":
            return np.mean(np.abs(A - Y))

        if self.l == "cross_entropy":
            return -np.mean(np.sum(Y * np.log(A + eps), axis=1))

        if self.l == "binary_cross_entropy":
            return -np.mean(Y * np.log(A + eps) + (1 - Y) * np.log(1 - A + eps))

        raise ValueError("invalid loss")

    def forward(self, X):

        A = X
        activations = [X]
        zs = []

        for layer in self.ls:
            Z = A @ layer["W"] + layer["B"]
            A = self.act(Z, layer["act"])
            zs.append(Z)
            activations.append(A)

        return activations, zs

    def save(self):

        if self.save_file is None:
            return

        with open(self.save_file, "wb") as f:
            pickle.dump({
                "ls": self.ls,
                "l": self.l,
                "classf": self.classf
            }, f)

    def load(self):

        with open(self.save_file, "rb") as f:
            data = pickle.load(f)

        self.ls = data["ls"]
        self.l = data["l"]
        self.classf = data["classf"]

    def train(self, X, Y, x=None, y=None, batch=32, epochs=10, lr=0.001):

        if not self.ls:
            raise ValueError("0 layers")

        l_a = self.ls[-1]["act"]

        if self.l == "cross_entropy" and l_a != "softmax":
            raise ValueError("cross_entropy without softmax")

        if self.l == "binary_cross_entropy" and l_a != "sigmoid":
            raise ValueError("binary_cross_entropy without sigmoid")

        for ly in self.ls[:-1]:
            if ly["act"] == "softmax":
                raise ValueError("softmax in non-output layer")

        n = len(X)

        for epoch in range(epochs):

            perm = np.random.permutation(n)
            Xs = X[perm]
            Ys = Y[perm]

            epoch_loss = 0.0
            batches = 0

            for start in range(0, n, batch):

                end = min(start + batch, n)

                currentX = Xs[start:end]
                currentY = Ys[start:end]

                m = len(currentX)

                activations, zs = self.forward(currentX)

                A = activations[-1]

                epoch_loss += self.loss(A, currentY)
                batches += 1

                if (self.l == "cross_entropy" or self.l == "binary_cross_entropy"):
                    dZ = A - currentY

                elif self.l == "mse":
                    dZ = ((A - currentY) * self.actd(zs[-1], l_a))

                elif self.l == "mae":
                    dZ = (np.sign(A - currentY) * self.actd(zs[-1], l_a))

                else:
                    raise ValueError("invalid loss")

                for j in reversed(range(len(self.ls))):

                    W_old = self.ls[j]["W"].copy()

                    A_prev = activations[j]

                    dW = (A_prev.T @ dZ) / m
                    dB = np.sum(dZ, axis=0, keepdims=True) / m

                    if j != 0:
                        dA_prev = dZ @ W_old.T

                    self.ls[j]["W"] -= lr * dW
                    self.ls[j]["B"] -= lr * dB

                    if j != 0:

                        prev_activation = self.ls[j - 1]["act"]

                        dZ = (dA_prev * self.actd(zs[j - 1], prev_activation))

            print(f"Epoch {epoch + 1}/{epochs} | " + f"Loss: {epoch_loss / batches:.6f}")

            self.save()

        if x is not None and y is not None:

            if self.classf:

                output = self.forward(x)[0][-1]

                if self.l == "binary_cross_entropy":

                    preds = (output >= 0.5).astype(int).reshape(-1)

                    if y.ndim > 1:
                        labels = y.reshape(-1)
                    else:
                        labels = y

                else:

                    preds = np.argmax(output, axis=1)

                    if y.ndim > 1:
                        labels = np.argmax(y, axis=1)
                    else:
                        labels = y

                print("Accuracy:", np.mean(preds == labels))

            else:

                pred = self.forward(x)[0][-1]

                mse = np.mean((pred - y) ** 2)

                print("Test MSE:", mse)
