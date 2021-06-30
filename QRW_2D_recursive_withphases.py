# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 13:42:00 2021

@author: User
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns; sns.set_theme()
plt.close('all')

#plt.rcParams.update({'font.size': 14})

def indexgrid(nshift,mshift,k):
    indexnarray = np.tile(np.arange(-k,k+1,1), (2*k+1,1)) + nshift*np.ones((2*k+1,2*k+1))
    indexmarray = np.transpose([np.flip(np.arange(-k,k+1,1))] * (2*k+1)) + mshift*np.ones((2*k+1,2*k+1))
    return indexnarray, indexmarray

# rightup = np.c_[np.zeros((2,(c[4].shape[1]+2))).T,np.r_[c[4],np.zeros((2,c[4].shape[0]))]]
# rightdown = np.c_[np.zeros((2,(c[4].shape[1]+2))).T,np.r_[np.zeros((2,c[4].shape[0])),c[4]]]
# leftup = np.c_[np.r_[c[4],np.zeros((2,c[4].shape[0]))],np.zeros((2,(c[4].shape[1]+2))).T]
# leftdown = np.c_[np.r_[np.zeros((2,c[4].shape[0])),c[4]],np.zeros((2,(c[4].shape[1]+2))).T]
        
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
    return probn/(4**k)

def plotprob(k, initc00, initc01, initc10, initc11, initposx, initposy):
    X = np.arange(-k+initposx,k+1+initposx,1)
    Y = np.arange(-k+initposy,k+1+initposy,1)
    X, Y = np.meshgrid(X,Y)
    Z = prob(k, initc00, initc01, initc10, initc11, initposx, initposy)
    Z = Z.astype(float)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    #plt.figure()
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
    return

def plotprobheatmap(k, initc00, initc01, initc10, initc11, initposx, initposy):
    plt.figure()
    data = prob(k, initc00, initc01, initc10, initc11, initposx, initposy)
    data = data.astype(float)
    sns.set(font_scale = 1.2)
    sns.heatmap(data, cmap = cm.coolwarm)
    plt.xticks(range(0, 2*k+1, 20), np.arange(-k,k+1,1)[::20])
    plt.yticks(range(0, 2*k+1, 20), np.flip(np.arange(-k,k+1,1))[::20])
    plt.xlabel("n")
    plt.ylabel("m")
    plt.tight_layout()
    plt.show
    return

### Functions that create matrices with distances/positions
def distance(k):
    dist = np.zeros(((2*k+1),(2*k+1)))
    x = np.arange(-k,k+1,1)
    y = np.flip(x)
    for i in range(2*k+1):
        for j in range(2*k+1):
            dist[i,j] = np.sqrt(x[i]**2+y[j]**2) 
    return dist

def posx(k):
    return np.tile(np.arange(-k,k+1,1), (2*k+1,1))

def posy(k):
    return np.transpose([np.flip(np.arange(-k,k+1,1))]*(2*k+1))

def posxy(k):
    return np.multiply(posx(k),posy(k))

### Functions for mean and variance of position x

def meanposx(k, initc00, initc01, initc10, initc11, initposx, initposy):
    probposx = np.multiply(posx(k), prob(k, initc00, initc01, initc10, initc11, initposx, initposy))
    return (np.sum(probposx))

def varposx(k, initc00, initc01, initc10, initc11, initposx, initposy):
    posxsquared = np.square(posx(k))
    s = np.multiply(posxsquared,prob(k, initc00, initc01, initc10, initc11, initposx, initposy))
    return np.sum(s) - meanposx(k, initc00, initc01, initc10, initc11, initposx, initposy)**2

def plotmeansx(k, initc00, initc01, initc10, initc11, initposx, initposy):
    meansx = np.zeros(k)
    for i in range(k):
        meansx[i] = meanposx(i+1, initc00, initc01, initc10, initc11, initposx, initposy)
    plt.figure()
    plt.title("Mean x coordinate of 2D walk plotted against number of steps with alpha*beta = "+str(round(alphabeta,2)))
    plt.xlabel("Number of steps")
    plt.ylabel("Mean x coordinate")
    plt.plot(np.arange(k),meansx)
    return

def plotvarx(k, initc00, initc01, initc10, initc11, initposx, initposy):
    variancesx = np.zeros(k)
    for i in range(k):
        variancesx[i] = varposx(i+1, initc00, initc01, initc10, initc11, initposx, initposy)
    #plt.figure()
    #plt.title("Variance of x coordinate of 2D walk plotted against number of steps with alpha*beta = "+str(round(alphabeta,2)))
    #plt.xlabel("Number of steps")
    #plt.ylabel("Variance of x coordinate")
    #plt.plot(np.arange(k),variancesx, label = "alpha*beta = "+str(round(alphabeta, 3)))
    return variancesx
    

### Functions for mean and variance of position y
def meanposy(k, initc00, initc01, initc10, initc11, initposx, initposy):
    probposy = np.multiply(posy(k), prob(k, initc00, initc01, initc10, initc11, initposx, initposy))
    return (np.sum(probposy))

def varposy(k, initc00, initc01, initc10, initc11, initposx, initposy):
    posysquared = np.square(posy(k))
    s = np.multiply(posysquared,prob(k, initc00, initc01, initc10, initc11, initposx, initposy))
    return np.sum(s) - meanposy(k, initc00, initc01, initc10, initc11, initposx, initposy)**2

def plotmeansy(k, initc00, initc01, initc10, initc11, initposx, initposy):
    meansy = np.zeros(k)
    for i in range(k):
        meansy[i] = meanposy(i+1, initc00, initc01, initc10, initc11, initposx, initposy)
    plt.figure()
    plt.title("Mean y coordinate of 2D walk plotted against number of steps with alpha*beta = "+str(round(alphabeta,2)))
    plt.xlabel("Number of steps")
    plt.ylabel("Mean y coordinate")
    plt.plot(np.arange(k),meansy)
    return

def plotvary(k, initc00, initc01, initc10, initc11, initposx, initposy):
    variancesy = np.zeros(k)
    for i in range(k):
        variancesy[i] = varposy(i+1, initc00, initc01, initc10, initc11, initposx, initposy)
    # plt.figure()
    # plt.title("Variance of y coordinate of 2D walk plotted against number of steps with alpha*beta = "+str(round(alphabeta,2)))
    # plt.xlabel("Number of steps")
    # plt.ylabel("Variance of y coordinate")
    plt.plot(np.arange(k),variancesy, label = "alpha*beta = "+str(round(alphabeta/np.pi, 3))+"*pi")




### number of steps
k = 100
alphabeta = 0
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

#cs = recursion(k)

# #Difference between prob distr of walks with different alpha beta
p = prob(k, initc00, initc01, initc10, initc11, initposx, initposy)
# alphabeta = np.pi/6+np.pi/2
# p2 = prob(k, initc00, initc01, initc10, initc11, initposx, initposy)

# diff = p-p2
# plt.figure()
# sns.heatmap(diff.astype(float),cmap = cm.coolwarm)

plotprob(k, initc00, initc01, initc10, initc11, initposx, initposy)

#plotprobheatmap(k, initc00, initc01, initc10, initc11, initposx, initposy)


#plots of variance for different values of alpha*beta
# plt.figure()
# #plt.title("Variance of x coordinate of 2D walk plotted against number of steps for different values of alpha*beta")
# plt.xlabel("t")
# plt.ylabel("$\sigma_x^2$")
# #for i in np.array([np.pi/2, np.pi/3, np.pi/4, np.pi/5, np.pi/8, np.pi/12, 0.1]):
# #for i in np.array([0, 0.001, 0.005, 0.01]):
# #alphabetas = np.array([np.pi/32, np.pi/16, np.pi*3/32, np.pi/8, np.pi*5/32, np.pi*3/16, np.pi*7/32, np.pi/4])
# #alphabetas = np.array([0, np.pi/24, np.pi/12, np.pi/8, np.pi/6, np.pi*5/24, np.pi/4]) + np.repeat(np.pi/4,7)
# #alphabetas = np.array([np.pi*2/3, np.pi*3/4, np.pi*5/6])
# alphabetas = np.array([0.001, 0.005, 0.01, 0.05, 0.1, 0.5])
# #alphabetas = np.array([np.pi/4])
# varx4 = np.zeros((len(alphabetas),k))
# j = 0
# #for i in np.array([0.01, 0.025, 0.05, 0.1, 0.5]):
# #for i in np.array([np.pi/2, np.pi/3, np.pi/4, np.pi/6, 2*np.pi/3]):
# for i in alphabetas:
#     alphabeta = i
#     varx4[j,:] = plotvarx(k, initc00, initc01, initc10, initc11, initposx, initposy)
#     plt.plot(np.arange(k),varx4[j,:], label = "alpha*beta = "+str(round(alphabeta/np.pi, 3))+"*$\pi$")
#     j+=1
# plt.plot(np.arange(0,100,1),np.arange(0,100,1),label="CRW")
# plt.legend()


#the figure with all fractions of pi alphabeta
# plt.figure();
# plt.plot(np.arange(1,101,1),varx[0], label=r"$\alpha\beta = 0$");
# plt.plot(np.arange(1,101,1),varx2[6], '--', label=r"$\alpha\beta = \pi/2$");
# plt.plot(np.arange(1,101,1),varx[4], label=r"$\alpha\beta = \pi/6$");
# plt.plot(np.arange(1,101,1),varx2[2], '--', label=r"$\alpha\beta = \pi/3$");
# plt.plot(np.arange(1,101,1),varx3[0], '-.', label=r"$\alpha\beta = 2\pi/3$");
# plt.plot(np.arange(1,101,1),varx3[2], ':', label=r"$\alpha\beta = 5\pi/6$");
# plt.plot(np.arange(1,101,1),varx[6], label=r"$\alpha\beta = \pi/4$");
# plt.plot(np.arange(1,101,1),varx3[1], '--', label=r"$\alpha\beta = 3\pi/4$")
# plt.plot(np.arange(0,100,1),np.arange(0,100,1),label="CRW");
# plt.xlabel("t", fontsize=14);
# plt.ylabel(r'$\sigma_x^2$', fontsize=14);
# plt.legend(fontsize = 14)

# # #the figure with log log scale of 3 fractions of pi
# plt.figure();
# plt.plot(np.arange(1,101,1),varx[0], label=r"$\alpha\beta = 0$");
# plt.plot(np.arange(1,101,1),varx[6], label=r"$\alpha\beta = \pi/4$");
# plt.plot(np.arange(1,101,1),varx[4], label=r"$\alpha\beta = \pi/6$");
# plt.plot(np.arange(0,100,1),np.arange(0,100,1),label="CRW");
# # plt.plot(np.log(np.arange(1,101,1)),np.log(varx[3]),label=r'$\alpha\beta = \pi/8$');
# plt.xlabel("t", fontsize=14);
# plt.ylabel(r'$\sigma_x^2$', fontsize=14)
# plt.legend(fontsize=14)

# #the figure with small fractions of pi
# #plt.plot(np.arange(1,101,1),varx[0],label=r'$\alpha\beta = 0$');
# plt.plot(np.arange(1,101,1),varx[1],label=r'$\alpha\beta = \pi/24$');
# plt.plot(np.arange(1,101,1),varx[2],label=r'$\alpha\beta = \pi/12$');
# plt.plot(np.arange(1,101,1),varx[3],label=r'$\alpha\beta = \pi/8$');
# #plt.plot(np.arange(1,101,1),varx[4],label=r'$\alpha\beta = \pi/6$');
# plt.plot(np.arange(1,101,1),varx[5],label=r'$\alpha\beta = 5\pi/24$');
# #plt.plot(np.arange(1,101,1),varx[6],label=r'$\alpha\beta = \pi/4$');
# plt.plot(np.arange(0,100,1),np.arange(0,100,1),label="CRW");
# plt.legend(fontsize=14);
# plt.xlabel("t", fontsize=14);
# plt.ylabel(r'$\sigma_x^2$', fontsize=14);
# plt.tight_layout()


# plt.plot(np.arange(1,101,1),varx2[6],label=r'$\alpha\beta = \pi/2$');
# plt.plot(np.arange(1,101,1),varx2[5],label=r'$\alpha\beta = 11\pi/24$');
# plt.plot(np.arange(1,101,1),varx2[4],label=r'$\alpha\beta = 5\pi/12$');
# plt.plot(np.arange(1,101,1),varx2[3],label=r'$\alpha\beta = 3\pi/8$');
# plt.plot(np.arange(1,101,1),varx2[2],label=r'$\alpha\beta = \pi/3$');
# plt.plot(np.arange(1,101,1),varx2[1],label=r'$\alpha\beta = 7\pi/24$');
# plt.plot(np.arange(1,101,1),varx2[0],label=r'$\alpha\beta = \pi/4$');
# plt.plot(np.arange(0,100,1),np.arange(0,100,1),label="CRW")
# plt.legend(fontsize=14);
# plt.xlabel("t", fontsize=14);
# plt.ylabel(r'$\sigma_x^2$', fontsize=14);
# plt.tight_layout()