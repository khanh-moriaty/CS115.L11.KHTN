from .activation import *

class ReLU(Activation):
    
    @staticmethod
    def forward(X):
        X = np.maximum(X, 0)
        return X
    
    @staticmethod
    def backward(X):
        G = np.ones_like(X)
        G[X < 0] = 0
        return G