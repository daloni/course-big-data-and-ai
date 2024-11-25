import numpy as np
import pickle

class NeuralNetwork:
    def __init__(self, layers, learning_rate=0.01):
        self.layers = layers
        self.learning_rate = learning_rate

    def forward_propagation(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def compute_loss(self, y_true, y_pred):
        m = y_true.shape[0]
        loss = -np.sum(y_true * np.log(y_pred + 1e-8)) / m
        return loss
    
    def backward_propagation(self, y_true, y_pred):
        dA = y_pred - y_true
        for layer in reversed(self.layers):
            dA = layer.backward(dA)

    def update_parameters(self):
        for layer in self.layers:
            layer.update_parameters(self.learning_rate)

    def fit(self, X, y, epochs=1000, batch_size=64, num_classes=10):
        for epoch in range(epochs):
            permutation = np.random.permutation(X.shape[0])
            X_shuffled = X[permutation]
            y_shuffled = y[permutation]
            
            for i in range(0, X.shape[0], batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]

                y_batch_one_hot = np.eye(num_classes, dtype=np.float32)[y_batch]
                y_pred = self.forward_propagation(X_batch)

                loss = self.compute_loss(y_batch_one_hot, y_pred)
                self.backward_propagation(y_batch_one_hot, y_pred)
                self.update_parameters()

            print(f"Época {epoch + 1}/{epochs}, Pérdida: {loss:.4f}")

    def predict(self, X):
        y_pred = self.forward_propagation(X)
        return np.argmax(y_pred, axis=1)

    def save_model(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self.layers, f)

    def load_model(self, file_path):
        with open(file_path, 'rb') as f:
            self.layers = pickle.load(f)
