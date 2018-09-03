# -*- coding: utf-8 -*-
"""
Created on Thur Apr  19 19:45:30 2018

@author: Kamil
"""

import numpy as np
import matplotlib.pyplot as plt
import numba
from scipy.stats import norm

# Creating a function to model Langevin equation using the Verlet
# algorithm with dissipative and noise terms. Based on Appendix C 
# from Chiara Liverani thesis
def norminv(percentile, mean, stddev):
    return norm.ppf(percentile, loc = mean, scale = stddev)

@numba.jit
def acceleration(position, mass):
    V1 = 1.5
    V2 = 0.75
    
    # Computing the value of acceleration
    accel = (2 * V1 * position - 4 * V2 * position ** 3) / mass
    return accel
    

@numba.jit
def model_langevin(t_max, time_step, q_init, p_init, T, gamma):
    
    # Specifying all of the physical parameters
    kb = 1
    mass = 1
    lamda = np.sqrt(2 * mass * T * kb * gamma)
    
    # Creating all of the necessary matrices
    time_points = np.arange(0, t_max, time_step, dtype = float)
    q_t = np.zeros(len(time_points), dtype = float)
    p_t = np.zeros(len(time_points), dtype = float)
    
    # initialising position and momentum
    q_t[0] = q_init
    p_t[0] = p_init
    
    # coefficients for q equation
    q_coeff1 = time_step * time_step / (2 * mass)
    q_coeff2 = lamda * (np.sqrt(time_step) ** 3) / (2 * mass)
    
    # coefficients for p equation
    p_coeff1 = 1 - gamma * time_step / mass
    p_coeff2 = time_step / (2 * mass)
    p_coeff3 = gamma * time_step * time_step / (2 * mass * mass)
    p_coeff4 = lamda * np.sqrt(time_step) / mass
    p_coeff5 = gamma * lamda * (np.sqrt(time_step) ** 3) / (2 * mass * mass)

    # implementing the method
    for i in range(len(time_points)):
        phi = np.random.normal(0,1)
        theta = np.random.normal(0,1)
        q_t[i+1] = q_t[i] + time_step * p_t[i] + q_coeff1 * (acceleration(q_t[i], mass) + gamma * p_t[i]) + \
                   q_coeff2 * (phi + theta / np.sqrt(3))
        p_t[i+1] = p_t[i] * p_coeff1 + p_coeff2 * (acceleration(q_t[i], mass) + acceleration(q_t[i+1], mass)) - \
                   p_coeff3 * (acceleration(q_t[i], mass) + gamma * p_t[i]) + p_coeff4 * phi - \
                   p_coeff5 * (phi + theta/np.sqrt(3))
    
    return([time_points, q_t, p_t])

# Creating a function to compute the transition rate using Eq. 2.66 found in 
# Chiara Liverani's PhD thesis. 

@numba.jit(parallel=True)
def transition_rate(no_of_realis, T, gamma, time_step, t_max):
    
    # Initialise arrays
    time_points = np.arange(0, t_max, time_step, dtype = float)
    q_t = np.zeros((len(time_points), no_of_realis), dtype = float)
    p_t = np.zeros((len(time_points), no_of_realis), dtype = float)
    trans_t = np.zeros(len(time_points), dtype = float)
    
    for i in range(no_of_realis):
        percentile = np.random.rand()
        p_init = norminv(percentile, 0, np.sqrt(T))
        q_init = 0
        model = model_langevin(t_max, time_step, q_init, p_init, T, gamma)
        p_t[:, i] = model[2]
        q_t[:, i] = model[1] 
        
    for i in range(no_of_realis):
        for j in range(len(time_points)):
            if q_t[j, i] >= 0:
                q_t[j, i] = 1
            else:
                q_t[j, i] = 0
    
    for i in range(len(time_points)):
        summation = p_t[0, :] * q_t[i, :]
        trans_t[i] = (1/no_of_realis) * np.sqrt(2 * np.pi/ T) * np.sum(summation)
    
    return([time_points, trans_t])

## Figure 2.6, page 29
#plt.figure()
#fig_26 = model_langevin(500, 0.001, 0, 0.1, 0.01, 0.001)
#plt.plot(fig_26[0], fig_26[1])
#
## Figure 2.7, page 30
#plt.figure()
#fig_27 = model_langevin(500, 0.001, 0, 0.1, 0.01, 0.1)
#plt.plot(fig_27[0], fig_27[1])
#
## Figure 2.8, page 31
#plt.figure()
#fig_28 = model_langevin(500, 0.001, 0, 0.1, 0.01, 0.5)
#plt.plot(fig_28[0], fig_28[1])
#
## Figure 2.14, page 36
#plt.figure()
#fig_214 = model_langevin(500, 0.001, 0, 0.1, 0.5, 0.5)
#plt.plot(fig_214[0], fig_214[1])
    
## Figure 2.4 page 27
#plt.figure()
#fig_24a = transition_rate(5000, 0.01, 0.001, 0.001, 50)
#plt.plot(fig_24a[0], fig_24a[1])
#
#plt.figure()
#fig_24b = transition_rate(5000, 0.01, 0.1, 0.001, 50)
#plt.plot(fig_24b[0], fig_24b[1])
#
#plt.figure()
#fig_24c = transition_rate(5000, 0.01, 0.5, 0.001, 50)
#plt.plot(fig_24c[0], fig_24c[1])
#
## Figure 2.13a page 35
#
#plt.figure()
#fig_213a = transition_rate(5000, 0.25, 0.001, 0.001, 50)
#plt.plot(fig_213a[0], fig_213a[1])