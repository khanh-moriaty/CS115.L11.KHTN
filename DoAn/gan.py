import numpy as np
import random
import matplotlib.pyplot as plt

from network.layer import *
from network.activation import *
from network.optimizer import *
from network.loss import *

class GAN():
    
    def __init__(self):
        self.G = []
        self.G.append(InputLayer(n=2))
        self.G.append(Layer(self.G[-1], n=1, activation=ReLU))
        self.G.append(Layer(self.G[-1], n=2, activation=Sigmoid))
        
        self.D = []
        self.D.append(InputLayer(n=2))
        self.D.append(Layer(self.D[-1], n=1, activation=ReLU))
        self.D.append(Layer(self.D[-1], n=1, activation=Sigmoid))
        
        self.GL = GANGNSLoss()
        self.GO = SGD(self.G, lr=0.2)
        self.DL = GANDLoss()
        self.DO = SGD(self.D)
        
        Z = np.linspace(0, 1, num=200)
        self.viz_bound = np.array(np.meshgrid(Z, Z)).reshape(2, -1).transpose()
        
    def gen_data(self):
        size = 100
        data = np.hstack((
            np.random.normal(1/4, 1/30, size=(size, 1)),
            np.random.normal(1/4, 1/30, size=(size, 1)),
        ))
        self.data = np.vstack((data,))
        size = 100
        data = np.hstack((
            np.random.normal(3/4, 1/30, size=(size, 1)),
            np.random.normal(3/4, 1/30, size=(size, 1)),
        ))
        self.data = np.vstack((self.data, data))
        fig = plt.figure()
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.scatter(self.data[:, 0], self.data[:, 1], c='red')
        plt.savefig('sample_data.jpg')
        
    def infer(self, n, i, verbose=False):
        Z = np.random.random((n, 2))
        X = self.G[0](Z)
        
        if verbose: print(X)
        
        fig = plt.figure()
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        Y = self.D[0](self.viz_bound).reshape(-1)
        Y = np.hstack((Y, [0, 1]))
        Z = np.vstack((self.viz_bound, [[0,0], [1,1]]))
        sc = plt.scatter(Z[:, 0], Z[:, 1], c=Y, cmap='RdYlBu')
        plt.colorbar(sc)
        plt.scatter(self.data[:, 0], self.data[:, 1], c='red')
        plt.scatter(X[:, 0], X[:, 1])
        plt.savefig('generated_data_{}.jpg'.format(i))
        
    def train(self, batch_size=2, d_loop=1, verbose=False, train_G=True):
        for _ in range(d_loop):
            # Train D
            Z = np.random.random((batch_size, 2))
            X_g = self.G[0](Z)
            X_d = self.data[np.random.choice(self.data.shape[0], batch_size, replace=True)]
            X = np.vstack((X_g, X_d))
            Y = np.ones((batch_size * 2,))
            Y[:batch_size] = 0
            Y = Y.reshape(-1, 1)
            Y_hat = self.D[0](X)
            if verbose:
                print(Y)
                print(Y_hat)
            loss_d = self.DL.compute(Y, Y_hat)
            self.D[0].backward(self.DL)
            self.DO.step()
            for G in self.D:
                G.G_W = 0
                G.G_b = 0
        
        if train_G:
            # Train G
            Z = np.random.random((batch_size, 2))
            X = self.G[0](Z)
            Y = np.zeros((batch_size,))
            Y = Y.reshape(-1, 1)
            Y_hat = self.D[0](X)
            loss_g = self.GL.compute(Y, Y_hat)
            self.D[0].backward(self.GL)
            self.G[0].backward(self.D[1])
            self.GO.step()
            for G in self.G:
                G.G_W = 0
                G.G_b = 0
            for G in self.D:
                G.G_W = 0
                G.G_b = 0
        
        
import time
    
if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)
    
    gan = GAN()
    gan.gen_data()
    
    total_time = 0
    max_iters = 50000
    for iter in range(1, max_iters+1):
        if iter % 1000 == 0 or (iter < 1000 and iter % 100 == 0):
            print('Iter {}/{} || G_loss: {:.5f} || D_loss: {:.5f} || avg_time: {:.5f} ||'.format(iter, max_iters, gan.GL.L, gan.DL.L, total_time/iter))
            Z = np.random.random((8, 2))
            X = gan.G[0](Z)
            print('X:\n', X)
            Y_hat = gan.D[0](X)
            print('Y_hat:\n', Y_hat)
            gan.infer(10000, iter)
        t = time.time()
        if iter == 500:
            for G in gan.G: 
                print('E:\n', G.E)
                print('G_W:\n', G.G_W)
                print('G_b:\n', G.G_b)
                print()
            print('D:')
            for G in gan.D: 
                print('E:\n', G.E)
                print('G_W:\n', G.G_W)
                print('G_b:\n', G.G_b)
                print()
        if iter < 1000:
            gan.train(d_loop=1)
        elif iter < 1200:
            gan.train(d_loop=200)
            gan.DO.lr = 0.64
            gan.GO.lr = 0.1
        elif iter < 2000:
            gan.train(d_loop=10)
            gan.GO.lr = 0.64
        elif iter < 3000:
            gan.train(d_loop=200)
            gan.GO.lr = 0.25
        elif iter < 4000:
            gan.train(d_loop=10)
            gan.GO.lr = 0.96
        elif iter < 5000:
            gan.train(d_loop=200)
        elif iter > 8000:
            gan.train(d_loop=1, train_G=False)
        else:
            gan.train(d_loop=10)
        total_time += time.time() - t
    
    # gan.infer(10, 0, verbose=True)
    
    # for i in range(100):
    #     gan.train()
    
    
    