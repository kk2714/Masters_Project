# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 18:03:56 2018

@author: Kamil
"""

from quantum_functions_library import *
import numpy as np
import matplotlib.pyplot as plt
import numba
    
# Integrating Lindblad equation directly to try and compute Tr(P_A rho(t))|rho_init = P_A \rho_eq

# Parameters of model
# v1 is the x^2 coefficient
# v2 is the x^4 coefficient
V1 = 1.5
V2 = 0.75
gamma = 0.1
T = 0.01
mass = 1
hbar = 1/(8.575*10 **(-34)/ (1.0545718 * 10**(-34)))
kb = 1
omega = 1
time_step = 0.0001
no_time_steps = 50000000
dim = 60
# Defining operators
x_operator = np.sqrt(hbar / (2 * mass * omega)) * (raising_operator(dim) + lowering_operator(dim)) 
p_operator = 1j * np.sqrt(hbar * mass * omega / 2) * (raising_operator(dim) - lowering_operator(dim)) 
lindblad = np.sqrt(2 * mass * kb * T) / hbar * x_operator + 1j / (2 * np.sqrt(2 * mass * kb * T)) * p_operator
h_sys = 1/(2 * mass) * p_operator * p_operator + V2 * x_operator * x_operator * x_operator * x_operator - V1 * x_operator * x_operator
hamiltonian = h_sys + gamma/2 * (x_operator * p_operator + p_operator * x_operator)
project_A = projection_A(dim, mass, omega, hbar)

# Define rho_equilirbium and then renormalise
rho_equilibrium = exp_matrix(-h_sys/(kb * T))
rho_equilibrium = rho_equilibrium / trace_matrix(rho_equilibrium)

# Define rho_initial and then renormalise
rho_init = project_A * rho_equilibrium
rho_init = rho_init / trace_matrix(rho_init)

tr_pa_rho = tr_rho_operator_euler(rho_init, project_A, hamiltonian, lindblad, hbar, time_step, no_time_steps)

plt.plot(np.arange(0, 10000 * 0.0001, 0.0001),tr_pa_rho[:10000])