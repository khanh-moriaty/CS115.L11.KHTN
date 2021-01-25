from .activation import *

# Actually LeakyRELU
class ReLU(Activation):
    
    @staticmethod
    def forward(X):
        X = np.maximum(X, 0.1*X)
        return X
    
    @staticmethod
    def backward(X):
        G = np.ones_like(X)
        G[X < 0] = 0.1
        return G