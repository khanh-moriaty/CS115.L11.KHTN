

class SGD():
    
    def __init__(self, layer_list, lr=0.01):
        self.layer_list = layer_list
        self.lr = lr
        
    
    def step(self):
        for layer in self.layer_list:
            layer.W -= self.lr * layer.G_W
            layer.b -= self.lr * layer.G_b