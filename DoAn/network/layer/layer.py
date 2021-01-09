import numpy as np
from network.activation import Activation

class Layer():
    
    def __init__(self, prev, n, activation=Activation):
        self.prev = prev
        self.prev.next = self
        self.next = None
        self.n = n
        self.activation = activation
        self.W = np.random.uniform(-1, 1, (self.prev.n, n))
        self.b = np.random.uniform(-1, 1, (n, 1))
        
    def __call__(self, X):
        return self.forward(X)
        
    def forward(self, X):
        assert len(X.shape) <= 2, "Invalid input shape ({} and {})".format(X.shape, self.n)
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        # Calculate Z and A
        self.Z = np.dot(self.W.T, X) + self.b
        self.A = self.activation.forward(self.Z)
        
        # Recursively call next layer
        if self.next:
            Y = self.next.forward(self.A)
        else:
            Y = self.A
        return Y
    
    def backward(self, loss):
        self.E = loss.E
        next_layer = None
        if self.next: next_layer = self.next
        elif isinstance(loss, Layer): next_layer = loss
        
        if next_layer:
            # If there exists a next layer, call backprop recursively
            if self.next:
                self.next.backward(loss)
                
            # Calculate error E(k) = W(k+1).E(k+1) * f'(Z(k))
            self.E = np.dot(next_layer.W, next_layer.E) * self.activation.backward(self.Z)
            
        self.G_W = np.dot(self.prev.A, self.E.T)
        self.G_b = np.sum(self.E, axis=1, keepdims=True)
        # self.G = np.clip(self.G, -1, 1)
        # print(A.shape, self.E.shape, self.G.shape, self.W.shape)