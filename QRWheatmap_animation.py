# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 15:48:33 2021

@author: User
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns
from matplotlib import cm
plt.close("all")

# figure, axis and plot elements
# fig = plt.figure()
# ax = plt.axes(xlim = (-100,100), ylim = (-100,100))
# heatmap, = sns.heatmap([])

# def init():
#     heatmap.set_data([])
#     return heatmap,

def indexgrid(nshift,mshift,k):
    indexnarray = np.tile(np.arange(-k,k+1,1), (2*k+1,1)) + nshift*np.ones((2*k+1,2*k+1))
    indexmarray = np.transpose([np.flip(np.arange(-k,k+1,1))] * (2*k+1)) + mshift*np.ones((2*k+1,2*k+1))
    return indexnarray, indexmarray
        
def shiftrightupphase(array, alphabeta, k):
    N, M = indexgrid(1,0,k)
    phases = np.multiply(np.exp(-1j*alphabeta*M),np.exp(1j*alphabeta*N))
    arrayphases = np.multiply(array,phases)
    return np.c_[np.zeros((2,(arrayphases.shape[1]+2))).T,np.r_[arrayphases,np.zeros((2,arrayphases.shape[0]))]]

def shiftrightdownphase(array, alphabeta, k):
    N, M = indexgrid(1,0,k)
    phases = np.multiply(np.exp(-1j*alphabeta*M),np.exp(-1j*alphabeta*N))
    arrayphases = np.multiply(array,phases)
    return np.c_[np.zeros((2,(arrayphases.shape[1]+2))).T,np.r_[np.zeros((2,arrayphases.shape[0])),arrayphases]]

def shiftleftupphase(array, alphabeta, k):
    N, M = indexgrid(-1,0,k)
    phases = np.multiply(np.exp(1j*alphabeta*M),np.exp(1j*alphabeta*N))
    arrayphases = np.multiply(array,phases)
    return np.c_[np.r_[arrayphases,np.zeros((2,arrayphases.shape[0]))],np.zeros((2,(arrayphases.shape[1]+2))).T]

def shiftleftdownphase(array, alphabeta, k):
    N, M = indexgrid(-1,0,k)
    phases = np.multiply(np.exp(1j*alphabeta*M),np.exp(-1j*alphabeta*N))
    arrayphases = np.multiply(array,phases)
    return np.c_[np.r_[np.zeros((2,arrayphases.shape[0])),arrayphases],np.zeros((2,(arrayphases.shape[1]+2))).T]

def recursion(k):
    if k == 1:
        c_pp00 = np.zeros((3,3),dtype='complex')
        c_pp00[0,2] = 1*np.exp(1j*alphabeta)
        c_pp01 = np.zeros((3,3))
        c_pp10 = np.zeros((3,3))
        c_pp11 = np.zeros((3,3))
        c_pm00 = np.zeros((3,3))
        c_pm01 = np.zeros((3,3),dtype='complex')
        c_pm01[2,2] = 1*np.exp(-1j*alphabeta)
        c_pm10 = np.zeros((3,3))
        c_pm11 = np.zeros((3,3))
        c_mp00 = np.zeros((3,3))
        c_mp01 = np.zeros((3,3))
        c_mp10 = np.zeros((3,3),dtype='complex')
        c_mp10[0,0] = 1*np.exp(-1j*alphabeta)
        c_mp11 = np.zeros((3,3))
        c_mm00 = np.zeros((3,3))
        c_mm01 = np.zeros((3,3))
        c_mm10 = np.zeros((3,3))
        c_mm11 = np.zeros((3,3),dtype='complex')
        c_mm11[2,0] = 1*np.exp(1j*alphabeta)
        return c_pp00, c_pp01, c_pp10, c_pp11, c_pm00, c_pm01, c_pm10, c_pm11, c_mp00, c_mp01, c_mp10, c_mp11, c_mm00, c_mm01, c_mm10, c_mm11
    else:
        c = recursion(k-1)
        c_pp00 = shiftrightupphase(c[0], alphabeta, k-1) + shiftrightupphase(c[4], alphabeta, k-1) + shiftrightupphase(c[8], alphabeta, k-1) + shiftrightupphase(c[12], alphabeta, k-1)
        c_pp01 = shiftrightupphase(c[1], alphabeta, k-1) + shiftrightupphase(c[5], alphabeta, k-1) + shiftrightupphase(c[9], alphabeta, k-1) + shiftrightupphase(c[13], alphabeta, k-1)
        c_pp10 = shiftrightupphase(c[2], alphabeta, k-1) + shiftrightupphase(c[6], alphabeta, k-1) + shiftrightupphase(c[10], alphabeta, k-1) + shiftrightupphase(c[14], alphabeta, k-1)
        c_pp11 = shiftrightupphase(c[3], alphabeta, k-1) + shiftrightupphase(c[7], alphabeta, k-1) + shiftrightupphase(c[11], alphabeta, k-1) + shiftrightupphase(c[15], alphabeta, k-1)
        c_pm00 = shiftrightdownphase(c[0], alphabeta, k-1) - shiftrightdownphase(c[4], alphabeta, k-1) + shiftrightdownphase(c[8], alphabeta, k-1) - shiftrightdownphase(c[12], alphabeta, k-1)
        c_pm01 = shiftrightdownphase(c[1], alphabeta, k-1) - shiftrightdownphase(c[5], alphabeta, k-1) + shiftrightdownphase(c[9], alphabeta, k-1) - shiftrightdownphase(c[13], alphabeta, k-1)
        c_pm10 = shiftrightdownphase(c[2], alphabeta, k-1) - shiftrightdownphase(c[6], alphabeta, k-1) + shiftrightdownphase(c[10], alphabeta, k-1) - shiftrightdownphase(c[14], alphabeta, k-1)
        c_pm11 = shiftrightdownphase(c[3], alphabeta, k-1) - shiftrightdownphase(c[7], alphabeta, k-1) + shiftrightdownphase(c[11], alphabeta, k-1) - shiftrightdownphase(c[15], alphabeta, k-1)
        c_mp00 = shiftleftupphase(c[0], alphabeta, k-1) + shiftleftupphase(c[4], alphabeta, k-1) - shiftleftupphase(c[8], alphabeta, k-1) - shiftleftupphase(c[12], alphabeta, k-1)
        c_mp01 = shiftleftupphase(c[1], alphabeta, k-1) + shiftleftupphase(c[5], alphabeta, k-1) - shiftleftupphase(c[9], alphabeta, k-1) - shiftleftupphase(c[13], alphabeta, k-1)
        c_mp10 = shiftleftupphase(c[2], alphabeta, k-1) + shiftleftupphase(c[6], alphabeta, k-1) - shiftleftupphase(c[10], alphabeta, k-1) - shiftleftupphase(c[14], alphabeta, k-1)
        c_mp11 = shiftleftupphase(c[3], alphabeta, k-1) + shiftleftupphase(c[7], alphabeta, k-1) - shiftleftupphase(c[11], alphabeta, k-1) - shiftleftupphase(c[15], alphabeta, k-1)
        c_mm00 = shiftleftdownphase(c[0], alphabeta, k-1) - shiftleftdownphase(c[4], alphabeta, k-1) - shiftleftdownphase(c[8], alphabeta, k-1) + shiftleftdownphase(c[12], alphabeta, k-1)
        c_mm01 = shiftleftdownphase(c[1], alphabeta, k-1) - shiftleftdownphase(c[5], alphabeta, k-1) - shiftleftdownphase(c[9], alphabeta, k-1) + shiftleftdownphase(c[13], alphabeta, k-1)
        c_mm10 = shiftleftdownphase(c[2], alphabeta, k-1) - shiftleftdownphase(c[6], alphabeta, k-1) - shiftleftdownphase(c[10], alphabeta, k-1) + shiftleftdownphase(c[14], alphabeta, k-1)
        c_mm11 = shiftleftdownphase(c[3], alphabeta, k-1) - shiftleftdownphase(c[7], alphabeta, k-1) - shiftleftdownphase(c[11], alphabeta, k-1) + shiftleftdownphase(c[15], alphabeta, k-1)
        return c_pp00, c_pp01, c_pp10, c_pp11, c_pm00, c_pm01, c_pm10, c_pm11, c_mp00, c_mp01, c_mp10, c_mp11, c_mm00, c_mm01, c_mm10, c_mm11

def prob(k, initc00, initc01, initc10, initc11, initposx, initposy):
    c = recursion(k)
    probn00 = np.square(np.abs(initc00*(c[0]+c[4]+c[8]+c[12])+initc01*(c[1]+c[5]+c[9]+c[13])+initc10*(c[2]+c[6]+c[10]+c[14])+initc11*(c[3]+c[7]+c[11]+c[15])))
    probn01 = np.square(np.abs(initc00*(c[0]-c[4]+c[8]-c[12])+initc01*(c[1]-c[5]+c[9]-c[13])+initc10*(c[2]-c[6]+c[10]-c[14])+initc11*(c[3]-c[7]+c[11]-c[15])))
    probn10 = np.square(np.abs(initc00*(c[0]+c[4]-c[8]-c[12])+initc01*(c[1]+c[5]-c[9]-c[13])+initc10*(c[2]+c[6]-c[10]-c[14])+initc11*(c[3]+c[7]-c[11]-c[15])))
    probn11 = np.square(np.abs(initc00*(c[0]-c[4]-c[8]+c[12])+initc01*(c[1]-c[5]-c[9]+c[13])+initc10*(c[2]-c[6]-c[10]+c[14])+initc11*(c[3]-c[7]-c[11]+c[15])))
    probn = probn00+probn01+probn10+probn11
    zero = np.zeros((201,201))
    zero[(100-k):(101+k),(100-k):(101+k)] = probn/(4**k)
    return zero


# def animate(i):
#     heatmap.set_data(prob(i, initc00, initc01, initc10, initc11, initposx, initposy))   
#     return heatmap,
#grid_kws = {'width_ratios': (0.9, 0.05), 'wspace': 0.2}
#fig, (ax, cbar_ax) = plt.subplots(1, 2, gridspec_kw = grid_kws, figsize = (12, 8))
fig, ax = plt.subplots()

def my_func(i):
    ax.cla()
    sns.heatmap(prob(i+1, initc00, initc01, initc10, initc11, initposx, initposy), cmap = cm.coolwarm, cbar = False)#, cbar_ax = cbar_ax)
    plt.xticks(range(0, 2*k+1, 10), np.arange(-k,k+1,1)[::10])
    plt.yticks(range(0, 2*k+1, 10), np.flip(np.arange(-k,k+1,1))[::10])

# zero = np.zeros((201,201))
# i = 0, zero[100,100]
# i = 1, zero [99:101,99:101]
# i = 2, zero[98:102,98:102]



### number of steps
k = 100
alphabeta = np.pi/2

### initial state (a*|0> + b*|1>)(cc*|0> + d*|1>)tensor |initposx, initposy>
#a = 1
a = 1/np.sqrt(2)
#b = 0
b = 1j/np.sqrt(2)
#cc = 1
cc = 1/np.sqrt(2)
#d = 0
d = 1j/np.sqrt(2)
initc00 = a*cc
initc01 = a*d
initc10 = b*cc
initc11 = b*d

initposx = 0
initposy = 0

#anim = animation.FuncAnimation(fig, animate, init_func=init, frames = 100, interval = 200, blit=True)
anim = FuncAnimation(fig = fig, func = my_func, frames = 100, interval = 200, blit = False)
anim.save('QRW2D_nophasefixed.mp4', fps=5) #extra_args=['-vcodec', 'libx264'])
plt.show()