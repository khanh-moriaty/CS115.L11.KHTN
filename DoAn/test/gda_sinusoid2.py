import math
import random
import numpy as np
import matplotlib.pyplot as plt

def f(x, y):
    return x*y

def dx(x, y):
    return y

def dy(x, y):
    return x

if __name__ == '__main__':
    
    bound = 1
    x = random.uniform(-bound, bound)
    y = random.uniform(-bound, bound)
    lr = 1e-3
    
    a = [[0, f(x, y)]]
    
    print(x, y, x+y)
    
    for i in range(20000):
        for _ in range(1):
            y = y + lr * dy(x, y)
        x = x - lr * dx(x, y)
        a.append([i, f(x, y)])
        # if (i+1) % 1000000 == 0:
        #     print(i+1, x, y, x+y, f(x, y))
    
    a = np.array(a)
    
    fig = plt.figure()
    plt.plot(a[:, 0], a[:, 1])
    plt.savefig('sinusoid2.jpg')