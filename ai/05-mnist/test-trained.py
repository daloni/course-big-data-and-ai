import NN
import NeuralNetwork
import numpy as np
import Dataloader
import random
import matplotlib.pyplot as plt

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

# Show example images
def show_images(images, title_texts, errors):
    cols = 7
    rows = int(len(images)/cols)
    plt.figure(figsize=(28, 28))
    index = 1    
    for x in zip(images, title_texts, errors):        
        image = x[0]        
        title_text = x[1]
        plt.subplot(rows, cols, index)
        color = plt.cm.Greens if x[2] else plt.cm.gray
        plt.imshow(image, cmap=color)
        if (title_text != ''):
            plt.title(title_text, fontsize = 15);        
        index += 1
    plt.show()

random_images = []
for i in range(0, 21):
    r = random.randint(0, len(x_train))
    image = x_train[r].reshape(1, -1)
    prediction = nn.predict(image)
    random_images.append((x_train[r], 'Image [' + str(r) + '] ' + str(y_train[r]) + '\n Predict ' + str(prediction[0]), y_train[r] != prediction[0]))

show_images(list(map(lambda x: x[0], random_images)), list(map(lambda x: x[1], random_images)), list(map(lambda x: x[2], random_images)))
