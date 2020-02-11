# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 18:57:12 2020

@author: Ming Cai
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time

font = {'family':'DejaVu Sans',
        'weight':'normal',
        'size'   : 30}

matplotlib.rc('font', **font)

# Bellman Value Iteration 

# Parameters 

alpha = 1/3
delta = 0.08
beta = 0.96

# Iteration 
n_iter = 100

# K range 0 to 10, accurate up to 1/100

accuracy = 100

k_min = 0 
k_max = 50
n = accuracy * (k_max - k_min) + 1

# space

k = np.linspace(k_min, k_max, n)
c = np.linspace(k_min, k_max, n)

def k_index(k):
    return np.array(np.around(k-k_min,decimals = np.int(np.log10(accuracy)))*accuracy, dtype = int)
    

# Production  

def f(k): # production function 
    return k**alpha

def mpk(k): # marginal product of capital 
    return alpha*(k**(alpha-1))

def u(c): # utility
    return np.log(c)

def mu(c): # marginal utility 
    return 1/c

def k_lom (c, k): # capital law of motion 
    return f(k) + (1-delta)*k - c 

# Value init 
V0 = np.zeros(n)

# Value function 
def Bellman(V0, k0):
    y = f(k0) + (1-delta)*k0
    index = k_index(k_lom(c, k0))
    j = np.argmax(np.where(c<=y, u(c)+beta*V0[index], -np.inf))
    return c[j], np.max(np.where(c<=y, u(c)+beta*V0[index], -np.inf))

def V_next(V_prev):
    V_now = np.zeros(n)
    pol = np.zeros(n)
    for i in range(n):
        pol[i], V_now[i] = Bellman(V_prev, k[i])
    return pol, V_now

V = np.ndarray([n_iter, n])
pol = np.zeros(n)
V[0] = V0


# Iteration 
begin = time.time()

for i in range(1, n_iter):
    a = time.time()
    pol, V[i] = V_next(V[i-1])
    b = time.time()
    print("Iteration: %d, time taken: %.2f seconds"%(i, b-a))

end = time.time()

print("Total time taken: %.2f seconds"%(end-begin))
# End Interation 

# Plot 
fig, ax = plt.subplots(1, 2)

fig.suptitle('Ramsey Model Value Iteration')

for i in range(n_iter-25, n_iter-1):
    ax[0].plot(k, V[i], linewidth = 3, color = 'blue')
    
ax[0].plot(k, V[n_iter-1], linewidth = 5, color = 'red')
ax[0].set_xlim([0,10])
ax[0].set_ylim([-5,10])
ax[0].set_title('Value Function')
ax[0].grid()

ax[1].plot(k, pol, color = 'cyan', linewidth = 5)
ax[1].set_xlim([0,8])
ax[1].set_ylim([0,3])
ax[1].set_title('Optimal Policy')
ax[1].grid()

plt.show()