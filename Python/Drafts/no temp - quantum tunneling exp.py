# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 17:54:14 2018

@author: Kamil
"""
from sse_chiara_2 import *
import matplotlib.pyplot as plt
import timeit
import time as time_clock
import numpy as np

### Looking at quantum tunneling without temperature effects by varying mass
  
# v1 is the x^2 coefficient
# v2 is the x^4 coefficient
V1 = 1.5
V2 = 0.75
gamma = 0.0
T = 0.01
mass = 0.1
hbar = 1/(8.575*10 **(-34)/ (1.0545718 * 10**(-34)))
kb = 1
omega = 1
time_step = 0.001
no_time_steps = 500000

# Convergence test
conv_test = convergence_function(V1, V2, gamma, T, mass, omega, time_step)
dim = conv_test[0]

# Initial conditions
percentile = np.random.rand()
p_init = 0.00 #norminv(percentile, 0, np.sqrt(T))
q_init = 1
init_wave = init_coherent_state(dim, p_init, q_init, mass, omega)

# Defining operators
x_operator = np.sqrt(hbar / (2 * mass * omega)) * (raising_operator(dim) + lowering_operator(dim)) 
p_operator = 1j * np.sqrt(hbar * mass * omega / 2) * (raising_operator(dim) - lowering_operator(dim)) 

# NO TEMPERATURE REGIME
lindblad = 0 * np.sqrt(4 * mass * kb * T) / hbar * x_operator + 0 * 1j / np.sqrt(4 * mass * kb * T) * p_operator
h_sys = 1/(2 * mass) * p_operator * p_operator + V2 * x_operator * x_operator * x_operator * x_operator - V1 * x_operator * x_operator

hamiltonian = h_sys + gamma/2 * (x_operator * p_operator + p_operator * x_operator)

# No temperature
time = np.arange(0, no_time_steps * time_step, time_step, dtype = float)
#unseeding code in a cheeky way
np.random.seed(int(time_clock.time()))
results = operator_time_average(init_wave, h_sys, lindblad, x_operator, gamma, time_step, no_time_steps, 1, "platen")
plt.errorbar(time, results[0], yerr=results[1])
plt.ylabel(r'$\langle q \rangle$')
plt.xlabel('t')
x1,x2,y1,y2 = plt.axis()
plt.axis((x1,x2,-2.0,2.0))