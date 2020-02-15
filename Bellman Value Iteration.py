# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 18:57:12 2020

@author: Ming Cai
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
import scipy.interpolate as itp

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
k_max = 50 # greater than maximum
n = accuracy * (k_max - k_min) + 1

# space

k = np.linspace(k_min, k_max, n)
c = np.linspace(k_min, k_max, n)
cmat, dum = np.meshgrid(k, c)

def k_index(k):
    index1 = np.around(k-k_min, decimals = np.int(np.log10(accuracy))) * accuracy
    iout = np.max([index1, np.zeros([n, n])], axis = 0)
    return np.array(iout, dtype = int)

def k_index_arr(k):
    index = np.around(k-k_min,decimals = np.int(np.log10(accuracy)))*accuracy
    iout = np.max([index, np.ones(n)], axis = 0)
    return np.array(iout, dtype = int)
    
# Production  

def f(k): # production function 
    return k**alpha

def mpk(k): # marginal product of capital 
    return alpha*(k**(alpha-1))

def u(c): # utility
    return np.log(c)

def u_inv(c):
    return np.exp(c)

def mu(c): # marginal utility 
    return 1/c

def y(k):
    return f(k) + (1-delta)*k

def k_lom (c, k): # capital law of motion 
    return y(k) - c

dum, ymat = np.meshgrid(k, y(k))
kmat = ymat - cmat

# Value init 
V0 = np.log(k)
V1 = np.zeros(n)
V2 = np.zeros(n)
V3 = np.zeros(n)

# Value function 
def Bellman(V0):
    index = k_index(kmat)
    return np.max(u(cmat)+beta*V0[index], axis = 1)

# Consumption Policy
k_c = np.linspace(k_min, k_max, 100)    

def c_pol(V0):
    epsilon = 1
    Vint = itp.interp1d(k, V0, fill_value = "extrapolate")
    slope = (Vint(k_c+epsilon) - Vint(k_c))
    c = mu(beta*slope)
    return c

# Iteration init 
V = np.ndarray([n_iter, n])
V[0] = V0

"""
#Test Line
V1 = Bellman(V0)
V2 = Bellman(V1)
V3 = Bellman(V2)
"""

# Iteration 
begin = time.time()

for i in range(1, n_iter):
    a = time.time()
    V[i] = Bellman(V[i-1])
    b = time.time()
    print("Iteration: %d, time taken: %.2f seconds"%(i, b-a))

copt = c_pol(V[n_iter-1])

end = time.time()

print("Total time taken: %.2f seconds"%(end-begin))
# End Interation

# Plot 
fig, ax = plt.subplots(1, 2)

fig.suptitle('Ramsey Model Value Iteration')

for i in range(2, n_iter-1):
    ax[0].plot(k, V[i], linewidth = 3, color = 'blue')
    
ax[0].plot(k, V[n_iter-1], linewidth = 5, color = 'red')
ax[0].set_xlim([0,8])
ax[0].set_ylim([-3,10])
ax[0].set_title('Value Function')
ax[0].grid()


ax[1].plot(k_c, copt, color = 'cyan', linewidth = 5)
ax[1].set_xlim([0,8])
ax[1].set_ylim([0,3])
ax[1].set_title('Optimal Policy Rough Prediction')
ax[1].grid()

# Functions 
fcopt = itp.interp1d(k_c, copt)
fValue = itp.interp1d(k, V[n_iter-1])

css_pred = fcopt(4.535)
vss_pred = fValue(4.535)

# Print Outputs 
print("\nSteady State Consumption is: %.4f, difference: %.4f"%(css_pred, css_pred-1.292))
print("\nSteady State Value is: %.4f, difference: %.4f"%(vss_pred, vss_pred-6.5591))
