# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 19:39:47 2021

@author: User
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns
from matplotlib import cm
import scipy.stats as stats
#plt.close('all')

def CRW1D(k):
    if k == 0:
        return np.array([1])
    else:
        c = CRW1D(k-1)
        right = np.concatenate((np.array([0,0]),c/2))
        left = np.concatenate((c/2, np.array([0,0])))
        return np.add(right,left)
    
def plotCRW1Dall(k, initpos):
    positions = np.arange(initpos-k, initpos+k+1, 1)
    plt.figure()
    plt.title("Probability distribution of 1D classical random walk with initial position $x_0$ = "+str(initpos)+" for t = "+str(k)+" steps")
    plt.xlabel("i")
    plt.ylabel("$p_t(i)$")
    plt.plot(positions, CRW1D(k))
    return

def plotCRW1Dnonzero(k, initpos):
    positions = np.arange(initpos-k, initpos+k+1, 2)
    #plt.figure()
    #plt.title("Probability distribution of 1D classical random walk with initial position $x_0$ = "+str(initpos)+" for t = "+str(k)+" steps")
    plt.xlabel("i", fontsize = 14)
    plt.ylabel("$P_{100}(i)$", fontsize = 14)
    plt.plot(positions, CRW1D(k)[::2],label="CRW")
    # mu = 0
    # sigma = np.sqrt(k)
    # plt.plot(positions, 2*stats.norm.pdf(positions, mu, sigma), color='red')
    # plt.plot(positions, 2*stats.norm.pdf(positions, mu, sigma)-CRW1D(k)[::2], color='green')
    return

def plotvarCRW1D(T):
    vars1D = np.zeros(T)
    for t in range(T):
        vars1D[t] = np.sum(np.multiply(np.square(np.arange(-t,t+1,1)),CRW1D(t)))
    #plt.figure()
    plt.xlabel("t",fontsize=14)
    plt.ylabel("$\sigma^2_x$", fontsize=14)
    #plt.title("Variance of 1D classical random walk with initial position $x_0$ as a function of number of steps t")
    plt.plot(np.arange(0,T,1),vars1D, label = "CRW")
    return
    
k = 100
initpos = 0


#plotCRW1Dall(k, initpos)
plotCRW1Dnonzero(k, initpos)
#plotvarCRW1D(100)

def shiftrightup(array):
    return np.c_[np.zeros((2,(array.shape[1]+2))).T,np.r_[array,np.zeros((2,array.shape[0]))]]

def shiftrightdown(array):
    return np.c_[np.zeros((2,(array.shape[1]+2))).T,np.r_[np.zeros((2,array.shape[0])),array]]

def shiftleftup(array):
    return np.c_[np.r_[array,np.zeros((2,array.shape[0]))],np.zeros((2,(array.shape[1]+2))).T]

def shiftleftdown(array):
    return np.c_[np.r_[np.zeros((2,array.shape[0])),array],np.zeros((2,(array.shape[1]+2))).T]

def CRW2D(k):
    if k==0:
        return np.array([[1]])
    else:
        c = CRW2D(k-1)
        rightup = shiftrightup(c/4)
        rightdown = shiftrightdown(c/4)
        leftup = shiftleftup(c/4)
        leftdown = shiftleftdown(c/4)
        return rightup+rightdown+leftup+leftdown
    
def plotheatmapCRW2D(k):
    plt.figure()
    data = CRW2D(k)
    data = data.astype(float)
    sns.heatmap(data, cmap = cm.coolwarm)
    plt.xticks(range(0, 2*k+1, 10), np.arange(-k,k+1,1)[::10])
    plt.yticks(range(0, 2*k+1, 10), np.flip(np.arange(-k,k+1,1))[::10])
    plt.show
    return

# plotheatmapCRW2D(k)

# 2D CRW from 2 1D CRWs
def CRW1Dx2(k):
    P = np.outer(CRW1D(k),CRW1D(k))
    plt.figure()
    data = P
    data = data.astype(float)
    sns.heatmap(data, cmap = cm.coolwarm)
    plt.xticks(range(0, 2*k+1, 10), np.arange(-k,k+1,1)[::10])
    plt.yticks(range(0, 2*k+1, 10), np.flip(np.arange(-k,k+1,1))[::10])
    plt.show
    return

# CRW1Dx2(k)


### Animation

#fig, ax = plt.subplots()

def CRW2Dani(k):
    zero = np.zeros((201,201))
    zero[(100-k):(101+k),(100-k):(101+k)] = CRW2D(k)
    return zero

def my_func(i):
    ax.cla()
    sns.heatmap(CRW2Dani(i), cmap = cm.coolwarm, cbar = False)#, cbar_ax = cbar_ax)
    plt.xticks(range(0, 2*k+1, 10), np.arange(-k,k+1,1)[::10])
    plt.yticks(range(0, 2*k+1, 10), np.flip(np.arange(-k,k+1,1))[::10])

# anim = FuncAnimation(fig = fig, func = my_func, frames = 100, interval = 200, blit = False)
# anim.save('CRW2D.mp4', fps=5, extra_args=['-vcodec', 'libx264'])
# plt.show()
