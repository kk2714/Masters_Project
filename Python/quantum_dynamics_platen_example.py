# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 18:01:00 2018

@author: Kamil
"""

from quantum_functions_library import *
import numpy as np
import matplotlib.pyplot as plt
import numba
import time as time_clock


## Recreating Chiara's stochastic simulations - Platen Working!!!! :)
    
# Parameters of model
    
# v1 is the x^2 coefficient
# v2 is the x^4 coefficient
V1 = 1.5
V2 = 0.75
gamma = 0.5
T = 0.01
mass = 1
hbar = 1/(8.575*10 **(-34)/ (1.0545718 * 10**(-34)))
kb = 1
omega = 1
time_step = 0.0001
no_time_steps = 5000000
dim = 50

# Initial conditions
percentile = np.random.rand()
p_init = norminv(percentile, 0, np.sqrt(T))
q_init = 1
init_wave = init_coherent_state(dim, p_init, q_init, mass, omega, hbar)

# Defining operators
x_operator = np.sqrt(hbar / (2 * mass * omega)) * (raising_operator(dim) + lowering_operator(dim)) 
p_operator = 1j * np.sqrt(hbar * mass * omega / 2) * (raising_operator(dim) - lowering_operator(dim)) 

# Temperature regime
lindblad = np.sqrt(4 * mass * kb * T * gamma) / hbar * x_operator + 1j * np.sqrt(gamma) / np.sqrt(4 * mass * kb * T) * p_operator
h_sys = 1/(2 * mass) * p_operator * p_operator + V2 * x_operator * x_operator * x_operator * x_operator - V1 * x_operator * x_operator

hamiltonian = h_sys + gamma/2 * (x_operator * p_operator + p_operator * x_operator)
time = np.arange(0, no_time_steps * time_step, time_step, dtype = float)
#unseeding code in a cheeky way
np.random.seed(int(time_clock.time()))
results = operator_time_average(init_wave, hamiltonian, lindblad, x_operator, hbar, time_step, no_time_steps, 1, "platen")
plt.errorbar(time, results[0], yerr=results[1])
plt.ylabel(r'$\langle q \rangle$')
plt.xlabel('t')
x1,x2,y1,y2 = plt.axis()
plt.axis((x1,x2,-2.0,2.0))