import NN
import NeuralNetwork
import numpy as np
import struct
import random
from sklearn.metrics import accuracy_score
from array import array

class MnistDataloader(object):
    def __init__(self, training_images_filepath,training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath

    def read_images_labels(self, images_filepath, labels_filepath):        
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = np.frombuffer(file.read(), dtype=np.uint8)
        
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            images = np.frombuffer(file.read(), dtype=np.uint8)
            images = images.reshape(size, rows, cols)

        return images, labels
            
    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train),(x_test, y_test)  

# Set file paths based on added MNIST Datasets
training_images_filepath = './datasets/train-images.idx3-ubyte'
training_labels_filepath = './datasets/train-labels.idx1-ubyte'
test_images_filepath = './datasets/t10k-images.idx3-ubyte'
test_labels_filepath = './datasets/t10k-labels.idx1-ubyte'

# Load MINST dataset
print('Loading MNIST dataset...')
mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

# Normalize
x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0

# Neuronal Network
layers = [
    NN.Input(28, 28),
    NN.Flatten(),
    NN.Dense(28*28, 1000),
    NN.LeakyReLu(),
    NN.Dense(1000, 100),
    NN.LeakyReLu(),
    NN.Dense(100, 10, activation="softmax"),
]

# Crear la red neuronal
nn = NeuralNetwork.NeuralNetwork(layers=layers, learning_rate=0.001)

nn.load_model("model.pickle")
exampleItemNumber = random.randint(0, len(x_train))
print("Number ", y_train[exampleItemNumber], " with index ", exampleItemNumber)

image = x_train[exampleItemNumber].reshape(1, -1)
prediction = nn.predict(image)

print("Predict ", prediction)
