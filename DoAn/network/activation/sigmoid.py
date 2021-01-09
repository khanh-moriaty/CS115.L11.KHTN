from .activation import *

class Sigmoid(Activation):
    
    @staticmethod
    def forward(X):
        Z = X.copy()
        mask = X < 0
        Z[mask] = np.exp(X[mask]) / (1 + np.exp(X[mask]))
        mask = X >= 0
        Z[mask] = 1 / (1 + np.exp(-X[mask]))
        return Z
    
    @staticmethod
    def backward(X):
        X = Sigmoid.forward(X)
        G = X * (1 - X)
        return G