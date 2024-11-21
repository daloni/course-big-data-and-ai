import NN
import NeuralNetwork
import numpy as np
import struct
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

# def batchify(data, label, batch_size):
#     for i in range(0, len(data), batch_size):
#         yield (data[i:i+batch_size], label[i:i+batch_size])

# def accuracy(y_pred, y):
#     return np.mean(np.argmax(y_pred, axis=1) == y)

# def train(x, y):
#     y_pred = my_first_NN.forward(x)
#     my_first_NN.backward(y_pred, y)
#     my_first_NN.update_parameters(0.01)
#     return my_first_NN.cost(y_pred, y)

# def test(x, y):
#     y_pred = my_first_NN.forward(x)
#     return accuracy(y_pred, y)

# ninputs = 28
# hl1 = 1000
# hl2 = 100
# nclass = 10

# batch_size = 1000
# epochs = 100
# epoch = 0

# layer_relu1 = NN.ReLu()
# layer_relu2 = NN.ReLu()

# dense1 = NN.Dense(ninputs, hl1)
# dense2 = NN.Dense(hl1, hl2)
# dense3 = NN.Dense(hl2, nclass)

# layer_input = NN.Input(ninputs, nclass)

# my_first_NN = NN.Model([
#     layer_input, 
#     dense1, 
#     NN.ReLu(), 
#     dense2, 
#     NN.ReLu(), 
#     dense3, 
# ])

# while epoch < epochs:
#     for batch in batchify(x_train, y_train, batch_size):
#         batch_data, batch_label = batch
#         cost = train(batch_data, batch_label)
#     accuracy = test(x_test, y_test)
#     if accuracy > 0.97:
#         NN.save_model()
#         break
#     epoch += 1
#     print("Epoch:", epoch, "Cost:", cost, "Acc:", accuracy)

# Definir la arquitectura de la red
layers = [
    NN.Input(28, 28),
    NN.Dense(28, 1000),
    NN.ReLu(),
    NN.Dense(1000, 100),
]

# Crear la red neuronal
nn = NeuralNetwork.NeuralNetwork(layers=layers, learning_rate=0.1)
# Entrenar la red neuronal en MNIST
nn.fit(x_train, y_train, epochs=100, batch_size=28*28)

# Evaluar la red neuronal
y_pred = nn.predict(x_test)
accuracy = accuracy_score(y_test.argmax(axis=1), y_pred)
print(f"Precisión en prueba: {accuracy:.4f}")
