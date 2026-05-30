class NeuralNetwork:

    def __init__(self, save_file=None, loss="cross_entropy"):

        self.layers = []
        self.save_file = save_file
        self.loss_name = loss

    def add_layer(self, inp, out, activation):

        W = np.random.uniform(
            low=-np.sqrt(6 / inp),
            high=np.sqrt(6 / inp),
            size=(inp, out)
        )

        B = np.zeros((1, out))

        self.layers.append({
            "W": W,
            "B": B,
            "activation": activation
        })

    def activate(self, Z, activation):

        if activation == "relu":
            return np.maximum(0, Z)

        if activation == "sigmoid":
            return 1 / (1 + np.exp(-Z))

        if activation == "tanh":
            return np.tanh(Z)

        if activation == "softmax":
            exp = np.exp(Z - np.max(Z, axis=1, keepdims=True))
            return exp / np.sum(exp, axis=1, keepdims=True)

    def activate_derivative(self, Z, activation):

        if activation == "relu":
            return (Z > 0).astype(float)

        if activation == "sigmoid":
            s = 1 / (1 + np.exp(-Z))
            return s * (1 - s)

        if activation == "tanh":
            return 1 - np.tanh(Z) ** 2

    def loss(self, A, Y):

        eps = 1e-8

        if self.loss_name == "mse":
            return np.mean((A - Y) ** 2) / 2

        if self.loss_name == "mae":
            return np.mean(np.abs(A - Y))

        if self.loss_name == "cross_entropy":
            return -np.mean(np.sum(Y * np.log(A + eps), axis=1))

        if self.loss_name == "binary_cross_entropy":
            return -np.mean(
                Y * np.log(A + eps) +
                (1 - Y) * np.log(1 - A + eps)
            )

        raise ValueError("Unknown loss")

    def loss_derivative(self, A, Y):

        eps = 1e-8

        if self.loss_name == "mse":
            return A - Y

        if self.loss_name == "mae":
            return np.sign(A - Y)

        if self.loss_name == "cross_entropy":
            return -(Y / (A + eps))

        if self.loss_name == "binary_cross_entropy":
            return (
                -(Y / (A + eps))
                + ((1 - Y) / (1 - A + eps))
            )

        raise ValueError("Unknown loss")

    def forward(self, X):

        A = X

        activations = [X]
        zs = []

        for layer in self.layers:

            Z = (A @ layer["W"]) + layer["B"]
            A = self.activate(Z, layer["activation"])

            zs.append(Z)
            activations.append(A)

        return activations, zs

    def save(self):

        if self.save_file is not None:

            with open(self.save_file, "wb") as f:
                pickle.dump(self.layers, f)

    def load(self):

        if self.save_file is not None:

            with open(self.save_file, "rb") as f:
                self.layers = pickle.load(f)

    def train(self, X, Y, x, y, batch, epochs, lr):

        classes = np.max(Y) + 1

        for epoch in range(epochs):

            epoch_loss = 0
            batches = 0

            for i in range(X.shape[0] // batch):

                currentX = X[i * batch:(i + 1) * batch]
                givenY = Y[i * batch:(i + 1) * batch]

                currentY = np.zeros((len(givenY), classes))
                currentY[np.arange(len(givenY)), givenY] = 1

                activations, zs = self.forward(currentX)

                A = activations[-1]

                loss = self.loss(A, currentY)

                epoch_loss += loss
                batches += 1

                dA = self.loss_derivative(A, currentY)

                last_activation = self.layers[-1]["activation"]

                if (
                    self.loss_name == "cross_entropy"
                    and last_activation == "softmax"
                ):
                    dZ = A - currentY

                elif (
                    self.loss_name == "binary_cross_entropy"
                    and last_activation == "sigmoid"
                ):
                    dZ = A - currentY

                else:
                    dZ = dA * self.activate_derivative(
                        zs[-1],
                        last_activation
                    )

                for j in reversed(range(len(self.layers))):

                    A_prev = activations[j]

                    dW = (A_prev.T @ dZ) / batch
                    dB = np.sum(dZ, axis=0, keepdims=True) / batch

                    if j != 0:
                        dA_prev = dZ @ self.layers[j]["W"].T

                    self.layers[j]["W"] -= lr * dW
                    self.layers[j]["B"] -= lr * dB

                    if j != 0:

                        dZ = dA_prev * self.activate_derivative(
                            zs[j - 1],
                            self.layers[j - 1]["activation"]
                        )

            print(
                f"Epoch {epoch + 1}/{epochs} | Loss: {epoch_loss / batches:.6f}"
            )

            self.save()

        correct = 0

        for inp, label in zip(x, y):
            inp = inp.reshape(1, -1)
            activations, _ = self.forward(inp)
            prediction = np.argmax(activations[-1])

            if prediction == label:
                correct += 1

        print("Accuracy:", correct / len(y))
