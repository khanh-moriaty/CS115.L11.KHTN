import numpy as np

from .loss import *

class GANGLoss(Loss):
    
    def compute(self, Y, Y_hat):
        assert Y.shape == Y_hat.shape, "Mismatch output size"
        assert np.all(Y == 0), "All input label must be synthetic data"
        # L = log-likelihood
        Y_hat[Y_hat == 0] = 1e-12
        Y_hat[Y_hat == 1] = 1 - 1e-12
        self.L = np.sum(np.log(1 - Y_hat)) / Y.shape[0]
        # E = nabla(L) = - 1/N * Y_hat
        self.E = - (Y_hat - Y) / Y.shape[0]
        self.E = self.E.T
        return self.L

class GANGNSLoss(Loss):
    
    def compute(self, Y, Y_hat):
        assert Y.shape == Y_hat.shape, "Mismatch output size"
        assert np.all(Y == 0), "All input label must be synthetic data"
        # L = negative log-likelihood
        Y_hat[Y_hat == 0] = 1e-12
        Y_hat[Y_hat == 1] = 1 - 1e-12
        self.L = - np.sum(np.log(Y_hat)) / Y.shape[0]
        # E = nabla(L) = 1/N * (1 - Y_hat)
        self.E = - (1 - Y_hat - Y) / Y.shape[0]
        self.E = self.E.T
        return self.L

class GANDLoss(Loss):
    
    def compute(self, Y, Y_hat):
        assert Y.shape == Y_hat.shape, "Mismatch output size"
        # L = negative log-likelihood
        Y_hat[Y_hat == 0] = 1e-12
        Y_hat[Y_hat == 1] = 1 - 1e-12
        self.L = - np.sum(Y * np.log(Y_hat) + (1 - Y) * np.log(1 - Y_hat)) / Y.shape[0]
        # self.L = - np.sum(Y * np.log(Y_hat)) / Y.shape[0]
        # E = nabla(L) = 1/N * Y_hat
        self.E = (Y_hat - Y) / Y.shape[0]
        self.E = self.E.T
        return self.L
