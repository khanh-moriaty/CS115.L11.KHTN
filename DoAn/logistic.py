import numpy as np
import random
import matplotlib.pyplot as plt

from network.layer import *
from network.activation import *
from network.optimizer import *
from network.loss import *

class Logistic():
    
    def __init__(self):
        self.D = []
        self.D.append(InputLayer(n=2))
        self.D.append(Layer(self.D[-1], n=1024, activation=Sigmoid))
        self.D.append(Layer(self.D[-1], n=1, activation=Sigmoid))
        
        self.DL = GANDLoss()
        self.DO = SGD(self.D)
        
        Z = np.linspace(0, 1, num=200)
        self.viz_bound = np.array(np.meshgrid(Z, Z)).reshape(2, -1).transpose()
        
    def gen_data(self):
        size = 800
        x = np.hstack((
            np.random.normal(1/2, 1/15, size=(size, 1)),
            np.random.normal(1/2, 1/15, size=(size, 1)),
        ))
        y = np.ones((size,))
        self.x = np.vstack((x,))
        self.y = np.hstack((y,))
        
        size = 100
        x = np.hstack((
            np.random.normal(1/6, 1/15, size=(size, 1)),
            np.random.normal(1/6, 1/15, size=(size, 1)),
        ))
        y = np.zeros((size,))
        self.x = np.vstack((self.x, x))
        self.y = np.hstack((self.y, y))
        
        size = 100
        x = np.hstack((
            np.random.normal(3/6, 1/15, size=(size, 1)),
            np.random.normal(1/6, 1/15, size=(size, 1)),
        ))
        y = np.zeros((size,))
        self.x = np.vstack((self.x, x))
        self.y = np.hstack((self.y, y))
        
        size = 100
        x = np.hstack((
            np.random.normal(5/6, 1/15, size=(size, 1)),
            np.random.normal(1/6, 1/15, size=(size, 1)),
        ))
        y = np.zeros((size,))
        self.x = np.vstack((self.x, x))
        self.y = np.hstack((self.y, y))
        
        size = 100
        x = np.hstack((
            np.random.normal(1/6, 1/15, size=(size, 1)),
            np.random.normal(3/6, 1/15, size=(size, 1)),
        ))
        y = np.zeros((size,))
        self.x = np.vstack((self.x, x))
        self.y = np.hstack((self.y, y))
        
        size = 100
        x = np.hstack((
            np.random.normal(5/6, 1/15, size=(size, 1)),
            np.random.normal(3/6, 1/15, size=(size, 1)),
        ))
        y = np.zeros((size,))
        self.x = np.vstack((self.x, x))
        self.y = np.hstack((self.y, y))
        
        size = 100
        x = np.hstack((
            np.random.normal(1/6, 1/15, size=(size, 1)),
            np.random.normal(5/6, 1/15, size=(size, 1)),
        ))
        y = np.zeros((size,))
        self.x = np.vstack((self.x, x))
        self.y = np.hstack((self.y, y))
        
        size = 100
        x = np.hstack((
            np.random.normal(3/6, 1/15, size=(size, 1)),
            np.random.normal(5/6, 1/15, size=(size, 1)),
        ))
        y = np.zeros((size,))
        self.x = np.vstack((self.x, x))
        self.y = np.hstack((self.y, y))
        
        size = 100
        x = np.hstack((
            np.random.normal(5/6, 1/15, size=(size, 1)),
            np.random.normal(5/6, 1/15, size=(size, 1)),
        ))
        y = np.zeros((size,))
        self.x = np.vstack((self.x, x))
        self.y = np.hstack((self.y, y))
        
        
        fig = plt.figure()
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.scatter(self.x[self.y == 0, 0], self.x[self.y == 0, 1], c='blue')
        plt.scatter(self.x[self.y == 1, 0], self.x[self.y == 1, 1], c='red')
        plt.savefig('sample_data.jpg')
        
    def infer(self, n, i, verbose=False):
        
        fig = plt.figure()
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        Y = self.D[0](self.viz_bound).reshape(-1)
        Y = np.hstack((Y, [0, 1]))
        Z = np.vstack((self.viz_bound, [[0,0], [1,1]]))
        sc = plt.scatter(Z[:, 0], Z[:, 1], c=Y, cmap='RdYlBu')
        plt.colorbar(sc)
        plt.scatter(self.x[self.y == 0, 0], self.x[self.y == 0, 1], c='blue')
        plt.scatter(self.x[self.y == 1, 0], self.x[self.y == 1, 1], c='red')
        plt.savefig('generated_data_{}.jpg'.format(i))
        
    def train(self, batch_size=8, d_loop=1, verbose=False):
        for _ in range(d_loop):
            # Train D
            Z = np.random.random((batch_size, 2))
            mask = np.random.choice(self.x.shape[0], batch_size, replace=True)
            X = self.x[mask]
            Y = self.y[mask]
            Y = Y.reshape(-1, 1)
            Y_hat = self.D[0](X)
            if verbose: 
                print(Y)
                print(Y_hat)
            loss_d = self.DL.compute(Y, Y_hat)
            self.D[0].backward(self.DL)
            self.DO.step()
            
        
import time
    
if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)
    
    gan = Logistic()
    gan.gen_data()
    
    # for _ in range(100):
    #     gan.train(verbose=True)
    #     for D in gan.D: 
    #         print(D.G_W)
    #         print(D.G_b)
    #     print(gan.DL.L, gan.DL.E)
    #     for D in gan.D: print(D.W)
        
    # gan.infer(10000, 0)
    
    total_time = 0
    max_iters = 100000
    for iter in range(1, max_iters+1):
        if iter % 5000 == 0:
            print('Iter {}/{} || D_loss: {:.5f} || avg_time: {:.5f} ||'.format(iter, max_iters, gan.DL.L, total_time/iter))
            gan.infer(10000, iter)
        t = time.time()
        if iter > 8000 and False:
            gan.train(verbose=True)
        else:
            gan.train()
        total_time += time.time() - t
    