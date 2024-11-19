import NN
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split

mnist = loadmat("../datasets/mnist-original.mat")
mnist_data = mnist["data"].T
mnist_label = mnist["label"][0].astype(int)

train_data, test_data,train_label, test_label = train_test_split(mnist_data, mnist_label, test_size=0.2)

def batchify(data, label, batch_size):
    for i in range(0, len(data), batch_size):
        yield (data[i:i+batch_size], label[i:i+batch_size])

def accuracy(y_hat, y):
    return np.mean(np.argmax(y_hat, axis=1) == y)

def train(x, y):
    y_hat = my_first_NN.forward(x)
    my_first_NN.backward(y_hat, y)
    my_first_NN.update_parameters(0.01)
    return my_first_NN.cost(y_hat, y)

def test(x, y):
    y_hat = my_first_NN.forward(x)
    return accuracy(y_hat, y)

ninputs = 784
hl1 = 1000
hl2 = 100
nclass = 10

batch_size = 1000
epochs = 100
epoch = 0

layer_relu1 = NN.ReLu()
layer_relu2 = NN.ReLu()

dense1 = NN.Dense(ninputs, hl1)
dense2 = NN.Dense(hl1, hl2)
dense3 = NN.Dense(hl2, nclass)

layer_input = NN.Input(ninputs, nclass)

my_first_NN = NN.Model([
    layer_input, 
    dense1, 
    NN.ReLu(), 
    dense2, 
    NN.ReLu(), 
    dense3, 
])

while epoch < epochs:
    for batch in batchify(train_data, train_label, batch_size):
        batch_data, batch_label = batch
        cost = train(batch_data, batch_label)
    accuracy = test(test_data, test_label)
    if accuracy > 0.97:
        NN.save_model()
        break
    epoch += 1
    print("Epoch:", epoch, "Cost:", cost, "Acc:", accuracy)
