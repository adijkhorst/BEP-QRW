# -*- coding: utf-8 -*-
"""
Created on Mon May 31 09:26:58 2021

@author: User
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
plt.close('all')

def recursion(k):
    if k == 1:
        c_plus0 = np.array([0,0,1])
        c_plus1 = np.zeros(3)
        c_min0 = np.zeros(3)
        c_min1 = np.array([1,0,0])
        return c_plus0, c_plus1, c_min0, c_min1
    else:
        c = recursion(k-1)
        c_plus0 = np.add(np.concatenate((np.array([0,0]),c[0])),np.concatenate((np.array([0,0]),c[1])))
        c_plus1 = np.add(np.concatenate((np.array([0,0]),c[2])),np.concatenate((np.array([0,0]),c[3])))
        c_min0 = np.subtract(np.concatenate((c[0],np.array([0,0]))),np.concatenate((c[1],np.array([0,0]))))
        c_min1 = np.subtract(np.concatenate((c[2],np.array([0,0]))),np.concatenate((c[3],np.array([0,0]))))
        return c_plus0, c_plus1, c_min0, c_min1
    
def prob(k, initc0, initc1, initpos):
    c = recursion(k)
    probn0 = np.square(np.abs(np.add(initc0*(np.add(c[0],c[2])),initc1*(np.add(c[1],c[3])))))
    probn1 = np.square(np.abs(np.add(initc0*(np.subtract(c[0],c[2])),initc1*(np.subtract(c[1],c[3])))))
    probn = np.add(probn0,probn1)
    return probn/(2**k)

def plotprob(k,initc0,initc1,initpos):
    positions = np.arange(-k+initpos,k+1+initpos,2)
    plt.figure()
    plt.plot(positions,prob(k,initc0,initc1,initpos)[::2],label="QRW")
    return

def meanpos(k, initc0, initc1, initpos):
    pos = np.arange(-k+initpos, k+1+initpos, 1)
    probpos = np.multiply(pos, prob(k, initc0, initc1, initpos))
    return np.sum(probpos)

def variancepos(k, initc0, initc1, initpos):
    possquared = np.square(np.arange(-k+initpos, k+1+initpos, 1))
    probpossquared = np.multiply(possquared,prob(k, initc0, initc1, initpos))
    return np.sum(probpossquared) - meanpos(k, initc0, initc1, initpos)

def plotQRW1Dvar(k, initc0, initc1, initpos):
    plt.figure()
    #plt.title("Variance of the QRW plotted against number of steps T")
    plt.xlabel("t", fontsize=14)
    plt.ylabel("$\sigma^2$", fontsize=14)
    var = np.zeros(k)
    for n in range(1,k):
        var[n] = variancepos(n, initc0, initc1, initpos)
    plt.plot(np.arange(1,k+1),var, label="QRW")
    return var

### number of steps
k = 100

### initial state initc0*|0> + initc1*|1> tensor |initpos>
#initc0 = 1
initc0 = 1/np.sqrt(2)
#initc1 = 0
initc1 = 1j/np.sqrt(2)
initpos = 0


plotprob(k, initc0, initc1, initpos)
#var1D = plotQRW1Dvar(k, initc0, initc1, initpos)
# p1d = prob(k, initc0, initc1, initpos)

### 2x 1D walk
# P = np.outer(p1d,p1d)

# X = np.arange(-k+initpos,k+1+initpos,1)
# Y = np.copy(X)
# X, Y = np.meshgrid(X,Y)
# Z = P.astype(float)
# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
#                     linewidth=0, antialiased=False)
# fig.colorbar(surf, shrink=0.5, aspect=5)
# #ax.plot_wireframe(X, Y, Z)
# plt.show()
