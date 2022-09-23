from typing import List
from Layer import Layer
import numpy as np
class NeuralNetwork:
    def __init__(self, X, Y, layers: List[Layer], learning_rate: float = 0.10) -> None:
        self.X = X
        self.Y = Y
        self.layers = layers
        self.lr = learning_rate

        # Number of training examples.
        self.m = Y.size
            
    def initialize(self):
        # Initializing weights and biases.
        for layer in self.layers:
            layer.initialize()

    def __one_hot_encode(self, Y):
        one_hot_Y = np.zeros((Y.size, Y.max() + 1))
        one_hot_Y[np.arange(Y.size), Y] = 1

        one_hot_Y = one_hot_Y.T
        return one_hot_Y

    def forward_prop(self) -> None:
        A = self.X
        for layer in self.layers:
            layer.Z = layer.W.dot(A) + layer.b
            
            layer.A = layer.activate(layer.Z)
            A = layer.A

    def back_prop(self) -> None:
        one_hot_Y = self.__one_hot_encode(self.Y)

        dZ_prec = None
        for i in reversed(range(len(self.layers))):
            if self.layers[i].is_output_layer:
                dZ = self.layers[i].A - one_hot_Y
            else:
                dZ = self.layers[i + 1].W.T.dot(dZ_prec) * self.layers[i].d_activate(self.layers[i].Z)

            if self.layers[i].is_first_layer:
                self.layers[i].dW = 1 / self.m * dZ.dot(self.X.T)
            else:
                self.layers[i].dW = 1 / self.m * dZ.dot(self.layers[i - 1].A.T)

            self.layers[i].db = 1 / self.m * np.sum(dZ)
            dZ_prec = dZ 

    def get_accuracy(self):
        predictions = np.argmax(self.layers[-1].A, 0)
        return np.sum(predictions == self.Y) / self.m

    def update_params(self) -> None:
        for layer in self.layers:
            layer.update_params(self.lr)

    def gradient_descent(self, iterations: int = 1000):
        self.initialize()

        for i in range(iterations):
            self.forward_prop()
            self.back_prop()
            self.update_params()

            if i % 10 == 0:
                print(f'Iteration: {i}')
                print(f'Accuracy: {self.get_accuracy()}')

    def predict(self, X):
        A = X
        for layer in self.layers:
            Z = layer.W.dot(A) + layer.b
            A = layer.activate(Z)

        #plt.imshow(X.reshape((28, 28)) * 255, interpolation = 'nearest')
        return np.argmax(A, 0)
