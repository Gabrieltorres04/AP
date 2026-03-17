import numpy as np

class NumpyDNN:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
            # input_size: n de palavras TF-IDF (1000)
            # hidden_size: n de neurónios na camada oculta
            # output_size: 5
        self.lr = learning_rate
        
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

    def relu(self, Z):
        return np.maximum(0, Z)

    def relu_derivative(self, Z):
        return Z > 0

    def softmax(self, Z):
        expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return expZ / np.sum(expZ, axis=1, keepdims=True)

    def compute_loss(self, Y_hat, Y):       # Categorical Cross-Entropy Loss
        m = Y.shape[0]
        epsilon = 1e-15
        loss = -np.sum(Y * np.log(Y_hat + epsilon)) / m
        return loss

    #* PROPAGAÇÃO

    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.relu(self.Z1)
        
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.softmax(self.Z2)
        return self.A2

    def backward(self, X, Y):
        m = X.shape[0]

        dZ2 = self.A2 - Y
        dW2 = np.dot(self.A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.relu_derivative(self.Z1)
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    #* TREINO E PREVISÃO
    def train(self, X, Y, epochs=1000):
        print("A iniciar o treino da rede neuronal")
        for i in range(epochs):
            Y_hat = self.forward(X)
            loss = self.compute_loss(Y_hat, Y)
            self.backward(X, Y)
            
            if i % 100 == 0 or i == epochs - 1:

                previsoes = np.argmax(Y_hat, axis=1)
                reais = np.argmax(Y, axis=1)
                accuracy = np.mean(previsoes == reais) * 100

                print(f"Época {i} | Loss: {loss:.4f} | Accuracy: {accuracy:.2f}%")
        print("Treino concluído")

    def predict(self, X):
        probs = self.forward(X)

            # Retorna o índice da classe com maior probabilidade
        return np.argmax(probs, axis=1)