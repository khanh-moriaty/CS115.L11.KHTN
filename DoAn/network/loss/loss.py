import numpy as np

class Loss():
    
    def __init__(self):
        self.L = 0
        self.E = 0
        
    def compute(self, Y, Y_hat):
        assert Y.shape == Y_hat.shape, "Mismatch output size"
        self.L = 0
        self.E = np.zeros_like(Y)
        return self.L