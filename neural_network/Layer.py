import numpy as np

class Layer:
    def __init__(self, 
                n_input: np.array, 
                n_neurons: np.array, 
                act_function: object, 
                d_act_function: object, 
                is_first_layer: bool = False, 
                is_output_layer: bool = False) -> None:
        self.weights_shape = (n_neurons, n_input)
        self.biases_shape = (n_neurons, 1)

        # activation function (ReLU, softmax, sigmoid etc.)
        self.act_function = act_function

        # derivative of the activation function
        self.d_act_function = d_act_function

        # bool: check if this is the first hidden layer
        self.is_first_layer = is_first_layer

        # bool: check if this is the last layer (output layer)
        self.is_output_layer = is_output_layer

        self.W, self.b = None, None
        self.dW, self.db = None, None

        self.Z, self.A = None, None

    def initialize(self) -> None:
        # -1 <= self.W <= 1
        self.W = np.random.rand(self.weights_shape[0], self.weights_shape[1]) - 0.5

        # -1 <= self.b <= 1
        self.b = np.random.rand(self.biases_shape[0], self.biases_shape[1]) - 0.5

    def activate(self, Z) -> np.array:
        return self.act_function(Z)

    def d_activate(self, Z) -> np.array:
        return self.d_act_function(Z)

    def update_params(self, lr) -> None:
        self.W -= lr * self.dW
        self.b -= lr * self.db