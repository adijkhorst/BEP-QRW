# -*- coding: utf-8 -*-
"""
Created on Wed May  5 11:54:02 2021

@author: User
"""
import numpy as np
import matplotlib.pyplot as plt
plt.close('all')

### Coin states
coin0 = np.array([1,0])  #|0>
coin1 = np.array([0,1])  #|1>

C00 = np.outer(coin0, coin0) # 0><0
C01 = np.outer(coin0, coin1) # 0><1
C10 = np.outer(coin1, coin0) # 1><0
C11 = np.outer(coin1, coin1) # 1><1

C = (C00+C01+C10-C11)/np.sqrt(2) #Hadamard coin
X = (C01+C10)

N = 100     #number of steps
initp = 0   #initial field state
P = initp + N   #possible number of positions after N steps
psi0 = np.kron(np.eye(P+1)[initp],coin0+1j*coin1)/np.sqrt(2) #symmetric
#psi0 = np.kron(np.eye(P+1)[initp],coin0)

positions = np.arange(0,P+1)

# build JC shift operator
JC = np.zeros((2*(P+1),2*(P+1)), dtype=complex)
JC[1:(2*P+1),1:(2*P+1)] = np.kron(np.eye(P), X)
JC[0,0] = 1
JC[2*P+1,2*P+1] = 1

#build bit flip operator
X_hat = np.kron(np.eye(P+1),X)

#build coin operator
C_hat = np.kron(np.eye(P+1),C)

#Total walk operator
U = C_hat.dot(X_hat).dot(JC)

#Analysis of eigenvalues of U
l, ev = np.linalg.eig(U)

#probability distribution after N steps
psiN = np.dot(np.linalg.matrix_power(U,N),psi0) #first P entries of psiN are coin0 positions and last P are coin1 positions

def QRW1D(psi0, N):
    psiN = np.dot(np.linalg.matrix_power(U,N),psi0)
    pos1 = np.copy(psiN[::2])
    pos2 = np.copy(psiN[1::2])
    prob = np.add(np.square(np.abs(pos1)),np.square(np.abs(pos2)))
    return prob

def QRW1Dpeaks(psi0, N):
    psiN = np.dot(np.linalg.matrix_power(U,N),psi0)
    pos1 = np.copy(psiN[::2])
    pos2 = np.copy(psiN[1::2])
    prob = np.add(np.square(np.abs(pos1)),np.square(np.abs(pos2)))
    peak1 = np.argmax(np.round(prob,5))
    probminpeak1 = np.copy(np.round(prob,5))
    probminpeak1[peak1] = 0
    return peak1, np.argmax(probminpeak1)

def plotpeaks(psi0, N):
    plt.figure()
    #plt.title("Location of peaks of the QRW plotted against number of steps t")
    plt.xlabel("t", fontsize = 14)
    plt.ylabel("index i of peak", fontsize=14)
    peaks1 = np.zeros(N+1)
    peaks2 = np.zeros(N+1)
    for n in range(N+1):
        peaks1[n] = QRW1Dpeaks(psi0, n)[0]
        peaks2[n] = QRW1Dpeaks(psi0, n)[1]
    plt.plot(np.arange(N+1),peaks1[:], label="left peak")
    plt.plot(np.arange(N+1),peaks2[:], label="right peak")
    plt.legend(fontsize = 14)
    return peaks1, peaks2

plotpeaks(psi0,N)

#plot probability distribution

def plotQRW1Dall(psi0, N):
    plt.figure()
    plt.xlabel("i", fontsize = 14)
    plt.ylabel(r"$p_{100}(i)$", fontsize = 14)
    return plt.plot(positions,QRW1D(psi0, N))

def plotQRW1Dnonzero(psi0, N):              #volgens mij werkt dit niet
    prob = QRW1D(psi0,N)
    prob[prob==0] = np.nan
    plt.figure()
    plt.title("Distribution of "+str(N)+" steps QRW with Hadamard coin and initial position symmetric (only even points plotted)")
    return plt.plot(positions,prob)

plotQRW1Dall(psi0, N)
#plotQRW1Dnonzero(psi0, N)


#mean and variance
def QRW1Dmean(psi0, N):
    return np.dot(QRW1D(psi0, N), positions)

def QRW1Dvar(psi0, N):
    return np.dot(QRW1D(psi0,N),np.square(positions))-QRW1Dmean(psi0, N)**2

def plotQRW1Dvar(psi0, N):
    plt.figure()
    plt.title("Variance of the QRW plotted against number of steps T")
    plt.xlabel("T")
    plt.ylabel("$\sigma^2$")
    var = np.zeros(N+1)
    for n in range(N+1):
        var[n] = QRW1Dvar(psi0, n)
    return plt.plot(np.arange(N+1),var)
        
#plotQRW1Dvar(psi0, N)

#limiting distribution
def limitingQRW1D(psi0, T):
    c = np.zeros(P+1)
    for t in range(T):
        c = np.add(c, QRW1D(psi0,t))
    return plt.plot(positions, c/T, label = "t ="+str(t+1))

# plt.figure()
# for T in np.arange(10,110,10):
#     limitingQRW1D(psi0,T)
# plt.legend()

#Check unitarity
#unitU = np.dot(U,U.conj().T)

