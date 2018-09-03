# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 17:52:05 2018

@author: Kamil
"""

import numpy as np
from math import factorial
from scipy.stats import norm
from numpy.polynomial.hermite import *
from scipy.integrate import quad
import math
from scipy import linalg

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
time_step = 0.001
no_time_steps = 50
dim = 80

# Initial conditions
percentile = np.random.rand()
p_init = 0.00 
q_init = 1
init_state = init_coherent_state(dim, p_init, q_init, mass, omega, hbar)

# Defining operators
x_operator = np.sqrt(hbar / (2 * mass * omega)) * (raising_operator(dim) + lowering_operator(dim)) 
p_operator = 1j * np.sqrt(hbar * mass * omega / 2) * (raising_operator(dim) - lowering_operator(dim)) 
lindblad = 0 * np.sqrt(2 * mass * kb * T) / hbar * x_operator + 0 * 1j / (2 * np.sqrt(2 * mass * kb * T)) * p_operator
h_sys = 1/(2 * mass) * p_operator * p_operator + V2 * x_operator * x_operator * x_operator * x_operator - V1 * x_operator * x_operator
hamiltonian = h_sys + gamma/2 * (x_operator * p_operator + p_operator * x_operator)
results = operator_time_average(init_state, hamiltonian, lindblad, x_operator, hbar, time_step, no_time_steps, 1, "rk")

init_state_str = 'hey'
results1 = [x_operator, p_operator]
T = 0.1
gamma = 0.2

##### Save all of the data
data_output_path = './Data/qm_sse_simulation_ind_path_' + init_state_str + \
                   '_t' + str(T).replace(".", "") + '_g' + str(gamma).replace(".", "") + '.npy'

np.save(data_output_path, results1)

##### Save all of the data
data_output_path = './Data/qm_sse_simulation_ind_path_' + init_state_str + \
                   '2_t' + str(T).replace(".", "") + '_g' + str(gamma).replace(".", "") + '.npy'

np.save(data_output_path, results)