import numpy as np
from network.activation import Activation
from .layer import Layer

class InputLayer(Layer):
    
    def __init__(self, n):
        self.next = None
        self.n = n
        self.W = 0
        self.b = 0
        self.E = 0
        self.G_W = 0
        self.G_b = 0
        
    def __call__(self, X):
        return self.forward(X)
        
    def forward(self, X):
        assert len(X.shape) <= 2 and X.shape[-1] == self.n, "Invalid input shape ({} and {})".format(X.shape, self.n)
            
        assert self.next, "Number of layers must be greater than 1."
            
        X = X.transpose()
        self.A = X
        Y = self.next.forward(X)
        Y = Y.transpose()
        return Y
    
    def backward(self, loss):
        assert self.next, "Number of layers must be greater than 1."
        self.next.backward(loss)