import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

def relu(x):
    return np.maximum(0, x)

def relu_prime(x):
    x[x > 0] = 1
    x[x < 0] = 0

    return x

# Mean squared erro
def mse(y_true, y_pred):
    return (np.mean((y_pred - y_true) ** 2))

def mse_derivative(y_true, y_pred):
    return (2 * (y_pred - y_true) / y_true.size)

def binary_cross_entropy_loss(y_true, y_pred):
    epsilon = 1e-15  # pequeno valor adicionado para evitar log(0)
    loss = - np.mean(y_true * np.log(y_pred + epsilon) + (1 - y_true) * np.log(1 - y_pred + epsilon))
    return loss

def binary_cross_entropy_loss_derivative(y_true, y_pred):
    return (y_pred - y_true) / (y_pred * (1 - y_pred))
    
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))


class Dense:
    '''
    Classe que cria a arquitetura da rede neural:
    entradas
    feat_size :: tamanho de entradas iniciais da arquitetura 1, 2, 3, .., n
    out_size :: geralmente 1 saida, ou entao, multicamadas
    '''
    def __init__(self, input_size, out_size):
        # Tamanho das dimensoes
        self.input_size = input_size
        self.out_size = out_size
        # np.random.normal(0, 1, 3 * 1) * np.sqrt(2 / 3)    array([ 0.40406558, -0.27438716,  0.58409914])
        # Criando um vetor de peso alinhado com a dimensao de entrada e saida
        self.weights = (np.random.normal(0, 1, input_size * out_size) * np.sqrt(2 / input_size)).reshape(input_size, out_size)
        #Avoid negative output
        self.bias = np.random.rand(1, out_size) - 0.5
    
    def forward(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward(self, output_derivative, lr=0.5):
        """
        output_derivative : derivada da saida
        lr: learning_rate
        """
        # erro = output - output_layer
        # Calcular o erro -> (w+1) = (w(n) * erro * lr)
        input_derivative = np.dot(output_derivative, self.weights.T)
        new_weight = np.dot(self.input.T.reshape(-1, 1), output_derivative)

        self.weights -= lr * new_weight
        self.bias -= lr * output_derivative

        return input_derivative

class Activation:
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)

        return self.output

    def backward(self, output_derivative, lr):
        return(self.activation_prime(self.input) * output_derivative)

class Network:
    '''
    Inicializando o modelo com a funcao de perda
    Encontrar o melhor valor da funcao de perda ou
    O melhor erro posivel do modelo
    '''
    def __init__(self, loss, loss_prime):
        self.layers = []
        self.loss = loss
        self.loss_prime = loss_prime

    def add(self, layer):
        self.layers.append(layer)

    def predict(self, input_data):

        result = []

        for i in range(len(input_data)):
            layer_output = input_data[i]

            for layer in self.layers:
                layer_output = layer.forward(layer_output)
            result.append(layer_output)

        return result

    def fit(self, x_train, y_train, epochs, lr):
        
        for e in range(epochs):
            error = 0

            for j in range(len(x_train)):

                layer_output = x_train[j]

                for layer in self.layers:
                    layer_output = layer.forward(layer_output)

                error += self.loss(y_train[j], layer_output)

                gradient = self.loss_prime(y_train[j], layer_output)

                for layer in reversed(self.layers):
                    gradient = layer.backward(gradient, lr)

            error /= len(x_train)
            
            print(f"Epoch {e + 1}/{epochs}  Erro={error}")
