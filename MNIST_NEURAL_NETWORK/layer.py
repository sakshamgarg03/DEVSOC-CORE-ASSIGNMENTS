import numpy as np

class Layer:
    def __init__(self, input_size, output_size, activation, activation_derivative):
        self.weights = np.random.randn(output_size, input_size) * 0.01
        self.biases = np.zeros((output_size, 1))
        self.activation = activation
        self.activation_derivative = activation_derivative
        self.input = None
        self.z = None
        self.a = None

    def forward(self, input):
        self.input = input
        self.z = np.dot(self.weights, input) + self.biases
        self.a = self.activation(self.z)
        return self.a
    
    def backward(self, dA, learning_rate,lambda_=0.0):
        if self.activation_derivative:
            dz = dA * self.activation_derivative(self.z)
        else:
            dz = dA  

        dw = np.dot(dz, self.input.T)
        db = dz
        d_input = np.dot(self.weights.T, dz)

        #using L2 regulation
        self.weights -= learning_rate *(dw +lambda_*self.weights)
        self.biases -= learning_rate * db

        return d_input
