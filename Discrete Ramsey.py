# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 11:29:08 2020

@author: Ming Cai
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 22:07:18 2020

@author: Ming Cai
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

font = {'family':'DejaVu Sans',
        'weight':'normal',
        'size'   : 30}

matplotlib.rc('font', **font)

# Discrete time Ramsey Model 

#path specific 

space = 20 # equally spaced consumption intervals

#parameters: 

d = 0.08 # depreciation 
b = 0.96 # subjective discount 
a = 1/3 # alpha in production function
k_init = 0.4535 #initial capital

def u(c):
    return np.log(c)
    
def m_u(c):
    return (1/c)
    
def f(k):
    return (k**a)
        
def mpk(k):
    return a*(k**(a-1))

def Ramsey(c0, k0):
    k1 = f(k0) + (1-d)*k0 - c0
    c1 = c0*b*(mpk(k0)+1-d)
    return np.array([c1,k1])

rep = 5
plot = False


def Optimal_Response(k_init, rep, plot=False):
    path = np.zeros([space,2,100])

    #feasible consumption init
    c_max = np.zeros(rep)
    c_min = np.zeros(rep)
    c_max[0] = f(k_init)+(1-d)*k_init
    c_min[0] = 0

    #path 
    for n in range(0, rep-1):
        #print("rep %d"%(n))
        c_max_temp = c_max[n]
        dist = (c_max[n]-c_min[n])/space
        #print("dist %.4f"%(dist))
        for i in range(1,space+1):
            path[i-1,0,0] = dist*i+c_min[n]
            path[i-1,1,0] = k_init
        for i in range(0,space):
            for j in range(1,100):
                path[i,:,j] = Ramsey(path[i,0,j-1],path[i,1,j-1])
                if path[i,1,j] <= 0:
                    c_max_temp = np.minimum(c_max_temp, path[i,0,0])
                    c_max[n+1] = c_max_temp
        c_min[n+1] = c_max[n+1]-(c_max[n]/space)
        
    #plot    
        if n >= 1 and plot == True:
            plt.plot(path[5,1,:],path[5,0,:],linewidth = 5, color = "blue")
            plt.plot(path[space-1,1,:],path[space-1,0,:],linewidth = 5, color = "blue")
            
    return np.array([c_min[rep-1],c_max[rep-1]])

#plot 
if plot == True:
    plt.xlabel("Capital",size = 30)
    plt.ylabel("Consumption",size = 30)
    plt.xlim([0,10])
    plt.ylim([0,3])
    plt.plot()

#Saddle Path 
n_pts = 120
k_min = 0.01
c_min = 0.01

sp = np.zeros([3, n_pts])
k_incr = 0.1

# loci
c_l = np.zeros([2,2])
k_l = np.zeros([2,100])
c_l[0,0] = 0
c_l[1,0] = 5


#saddle path 
for i in range(0,n_pts):
    sp[0,i] = k_min + i*k_incr

for i in range(0,n_pts):
    sp[1:3,i] = Optimal_Response(sp[0,i],rep) 


#loci
def k_locus(k):
    return f(k)-d*k

def c_locus(k):
    return ((1/a)*(1/b-1+d))**(1/(a-1))

for i in range(0,100):
    k_l[1,i] = k_min + i*k_incr
    k_l[0,i] = k_locus(k_l[1,i])

for i in range(0,2):
    c_l[i,1] = c_locus(c_l[i,0])
    
Optimal_Response(0.01, rep, plot = True)
Optimal_Response(10, rep, plot = True)
plt.plot(c_l[:,1], c_l[:,0], linewidth = 5, color = "black")
plt.plot(k_l[1,:],k_l[0,:], linewidth = 5, color = "black")

#plt.plot(sp[0,:],sp[1,:], linewidth = 5, color = "orange")
plt.plot(sp[0,:],sp[2,:], linewidth = 5, color = "red")

#steady state
plt.plot(4.535, 1.292, 'ro', markersize=15)

plt.xlabel("Capital",size = 30)
plt.ylabel("Consumption",size = 30)
plt.xlim([0,10])
plt.ylim([0,3])
plt.plot()
