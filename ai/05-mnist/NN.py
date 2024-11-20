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

class Dense(Layer):
    def __init__(self, ninputs, noutputs):
        self.W = np.random.randn(ninputs, noutputs) * 0.01
        self.B = np.zeros((1, noutputs))

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.Z = x
        self.bs = x.shape[0]
        return x @ self.W + self.B

    def backward(self, error) -> np.ndarray:
        self.dW = self.Z.T @ error
        self.dB = np.sum(error, axis=0, keepdims=True)
        return error @ self.W.T

    def update_parameters(self, learning_rate):
        self.W = self.W - learning_rate / self.bs * self.dW
        self.B = self.B - learning_rate / self.bs * self.dB

class ReLu(Activation):
    def __init__(self):
        self.act_function = self.activate

    def forward(self, input: np.ndarray) -> np.ndarray:
        self.Z = self.activate(input)
        return self.Z

    def backward(self, error) -> np.ndarray:
        return error*self.activate(self.Z, derivative=True)   

    def activate(self, input: np.ndarray, derivative = False) -> np.ndarray:
        if derivative:
            y = np.ones(input.shape)
            y[input<=0] = 0
            return y
        else:
            return np.maximum(0, input)

    def update_parameters(self, learning_rate):
        pass 

class Model():
    def __init__(self, layers: list):
        self.layers = layers

    def forward(self, input: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            input = layer.forward(input)
        return input

    def backward(self, y_hat, y) -> np.ndarray:
        error = self.cost(y_hat, y)
        for layer in reversed(self.layers):
            error = layer.backward(error)
        return error

    def cost(self, y_hat, y):
        return (self.one_hot(y)-y_hat)*2

    def update_parameters(self, learning_rate=0.03):
        for layer in self.layers:
            layer.update_parameters(learning_rate)

    def save_model(self):
        file = open("model", "w")
        file.write(str(self.layers))

    def one_hot(self, labels):
        y = np.eye(10)[labels]
        return y
