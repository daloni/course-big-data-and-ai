import NN
import NeuralNetwork
import Dataloader
import numpy as np
from sklearn.metrics import accuracy_score

# Set file paths based on added MNIST Datasets
training_images_filepath = './datasets/train-images.idx3-ubyte'
training_labels_filepath = './datasets/train-labels.idx1-ubyte'
test_images_filepath = './datasets/t10k-images.idx3-ubyte'
test_labels_filepath = './datasets/t10k-labels.idx1-ubyte'

# Load MINST dataset
print('Loading MNIST dataset...')
mnist_dataloader = Dataloader.Dataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

# Normalize
# x_train = x_train.astype(np.float32) / 255.0
# x_test = x_test.astype(np.float32) / 255.0
x_train = np.where(x_train > 1, 1, 0)
x_test = np.where(x_test > 1, 1, 0)

layers = [
    NN.Input(28, 28),
    NN.Flatten(),
    NN.Dense(28*28, 1000),
    NN.LeakyReLu(),
    NN.Dense(1000, 50),
    NN.LeakyReLu(),
    NN.Dense(50, 200),
    NN.LeakyReLu(),
    NN.Dense(200, 30),
    NN.LeakyReLu(),
    NN.Dense(30, 10, activation="softmax"),
]

# Create Neuronal Network
nn = NeuralNetwork.NeuralNetwork(layers=layers, learning_rate=0.01)
nn.fit(x_train, y_train, epochs=100, batch_size=500)

y_pred = nn.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Precision in test: {accuracy:.4f}")

nn.save_model("model.pickle")
