from abc import ABC as abc, abstractmethod
import numpy as np

LEARING_RATE = 0.1

class Layer(abc):
    @abstractmethod
    def __init__(self, ninputs, noutputs):
        pass

    @abstractmethod
    def forward(self, input: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, error) -> np.ndarray:
        pass

    @abstractmethod
    def update_parameters(self, learning_rate):
        pass

class Input(Layer):
    def __init__(self, ninputs, noutputs):
        pass

    def forward(self, input: np.ndarray) -> np.ndarray:
        return input

    def backward(self, error):
        return error

    def update_parameters(self, learning_rate):
        pass

class Activation(Layer):
    def __init__(self, act_function):
        self.act_function = act_function

    def forward(self, input: np.ndarray) -> np.ndarray:
        return self.act_function(input)

class Flatten(Layer):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def forward(self, X):
        self.input_shape = X.shape
        return X.reshape(X.shape[0], self.num_classes)

    def backward(self, dA):
        return dA.reshape(self.input_shape)

    def update_parameters(self, learning_rate):
        pass

class Dense(Layer):
    def __init__(self, ninputs, noutputs, activation=None, initialization=1):
        if initialization != 1 and initialization != 2:
            initialization = 1

        self.activation = activation
        self.W = np.random.randn(ninputs, noutputs) * np.sqrt(initialization / ninputs)
        self.B = np.zeros((1, noutputs))

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.Z = x
        self.bs = x.shape[0]
        output = x @ self.W + self.B

        if self.activation == "softmax":
            exp_values = np.exp(output - np.max(output, axis=1, keepdims=True))
            return exp_values / np.sum(exp_values, axis=1, keepdims=True)
        elif self.activation == "relu":
            return np.maximum(0, output)

        return output

    def backward(self, error) -> np.ndarray:
        self.dW = self.Z.T @ error
        self.dB = np.sum(error, axis=0, keepdims=True)
        return error @ self.W.T

    def update_parameters(self, learning_rate):
        self.W = self.W - learning_rate / self.bs * self.dW
        self.B = self.B - learning_rate / self.bs * self.dB

class LeakyReLu(Activation):
    def __init__(self, alpha=0.01):
        self.act_function = self.activate
        self.alpha = alpha

    def forward(self, input: np.ndarray) -> np.ndarray:
        self.Z = input
        return self.activate(input)

    def backward(self, error) -> np.ndarray:
        return error * self.activate(self.Z, derivative=True)   

    def activate(self, input: np.ndarray, derivative=False) -> np.ndarray:
        if derivative:
            return np.where(self.Z > 0, 1, self.alpha)
        else:
            return np.where(input > 0, input, self.alpha * input)

    def update_parameters(self, learning_rate):
        pass 
