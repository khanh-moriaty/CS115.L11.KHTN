import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def f(x, y):
    return np.sin(x + y)

def dx(x, y):
    return np.cos(x + y)

def dy(x, y):
    return np.cos(x + y)

def plot_swarm(a, out_file):
    fig = plt.figure()
    # fig.set_size_inches(15, 10, True)
    # fig.set_tight_layout(True)
    # fig.suptitle(func_name.upper(), size='xx-large')
    
    lo = (-10, -10)
    hi = (+10, +10)
    
    ax_swarm = fig.add_subplot(111, aspect='equal')
    ax_swarm.set_xlim(lo[0], hi[0])
    ax_swarm.set_ylim(lo[1], hi[1])
    ax_swarm.set_xlabel('Swarm Population', size='xx-large')
    
    INTERVALS = 1000
    
    xlist = np.linspace(lo[0], hi[0], INTERVALS)
    ylist = np.linspace(lo[1], hi[1], INTERVALS)
    
    X, Y = np.meshgrid(xlist, ylist)
    print(X.shape)
    Z = f(X, Y)
    
    cp = ax_swarm.contourf(X, Y, Z, levels=100, cmap=plt.cm.bone)
    # fig.colorbar(cp)
    scatter_swarm = ax_swarm.scatter(a[:1, 0], a[:1, 1], c='red', s=20)
    
    def update(i):
        print(i)
        i = np.clip(i, 0, len(a)-1)
        label = "Generation {:02d}".format(i)
        fig.suptitle(label, size='xx-large')
        
        scatter_swarm.set_offsets(a[:i+1])
        return fig, scatter_swarm
    
    anim = FuncAnimation(fig, update, frames=np.arange(1, 3), interval=100)
    anim.save(out_file, dpi=150, writer='imagemagick')

if __name__ == '__main__':
    
    bound = 1
    x = random.uniform(-bound, bound)
    y = random.uniform(-bound, bound)
    lr = 1e-3
    
    a = [[0, x+y]]
    
    print(x, y, x+y)
    
    for i in range(2000000):
        for _ in range(1):
            y = y + lr * dy(x, y)
        x = x - lr * dx(x, y)
        a.append([i, x+y])
        if (i+1) % 1000000 == 0:
            print(i+1, x, y, x+y, f(x, y))
    
    a = np.array(a)
    
    fig = plt.figure()
    plt.plot(a[:, 0], a[:, 1])
    plt.savefig('sinusoid1.jpg')
    