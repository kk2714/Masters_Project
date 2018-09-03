# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 17:59:18 2018

@author: Kamil
"""

from quantum_functions_library import *
import numpy as np
import matplotlib.pyplot as plt
import time as time_clock

## Numerical convergence test! :)
    
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
time_step = 0.001
no_time_steps = 1000
dim = 50

# Initial conditions
percentile = np.random.rand()
p_init = norminv(percentile, 0, np.sqrt(T))
q_init = 1
init_wave = init_coherent_state(dim, p_init, q_init, mass, omega, hbar)
#
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

rho_operator_time = int_lindblad_liouv_euler(init_wave, hamiltonian, lindblad, hbar, 0.00001, 100000)
position = []
for j in range(rho_operator_time.shape[2]):
    position_t = 0
    for k in range(dim):
        sho_basis = np.matrix(np.zeros((1, dim), dtype = complex), dtype = complex).T
        sho_basis[k, 0] = 1
        position_t += sho_basis.getH() * x_operator * rho_operator_time[:,:,j] * sho_basis
    position.append(position_t[0,0])

results_platen = operator_time_average(init_wave, hamiltonian, lindblad, x_operator, hbar, time_step, no_time_steps, 1500, "platen")
results_rk = operator_time_average(init_wave, hamiltonian, lindblad, x_operator, hbar, time_step, no_time_steps, 1500, "rk")

# Plots in following order: Lindblad, RK, Platen
plt.plot(time, position[::100])
plt.errorbar(time, results_rk[0], yerr=results_rk[1])
plt.errorbar(time, results_platen[0], yerr=results_platen[1])
plt.ylabel(r'$\langle q \rangle$')
plt.xlabel('t')
plt.legend(('Lindblad', 'Runge-Kutta', 'Platen'),
           shadow=True, loc=(0.80, 0.20))
plt.savefig('num_conv.png')