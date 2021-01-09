import numpy as np

class Activation():
    
    @staticmethod
    def forward(X):
        return X
        
    @staticmethod
    def backward(X):
        return np.ones_like(X)
    