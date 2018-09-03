# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 17:52:05 2018

@author: Kamil
"""
from quantum_functions_library import *
import matplotlib.pyplot as plt
import time as time_clock
import numpy as np
import sys, os
import scipy.io

### For clearing up any file paths
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

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
no_time_steps = 500000
dim = 80

# Initial conditions
percentile = np.random.rand()
p_init = 0.00 
q_init = 1
init_state = init_coherent_state(dim, p_init, q_init, mass, omega, hbar)

# Defining operators
x_operator = np.sqrt(hbar / (2 * mass * omega)) * (raising_operator(dim) + lowering_operator(dim)) 
p_operator = 1j * np.sqrt(hbar * mass * omega / 2) * (raising_operator(dim) - lowering_operator(dim)) 

# NO TEMPERATURE REGIME
lindblad = 0 * np.sqrt(2 * mass * kb * T) / hbar * x_operator + 0 * 1j / (2 * np.sqrt(2 * mass * kb * T)) * p_operator
h_sys = 1/(2 * mass) * p_operator * p_operator + V2 * x_operator * x_operator * x_operator * x_operator - V1 * x_operator * x_operator

hamiltonian = h_sys + gamma/2 * (x_operator * p_operator + p_operator * x_operator)

# No temperature
time = np.arange(0, no_time_steps * time_step, time_step, dtype = float)
#unseeding code in a cheeky way
np.random.seed(int(time_clock.time()))
results = operator_time_average(init_state, hamiltonian, lindblad, x_operator, hbar, time_step, no_time_steps, 1, "rk")

init_state_str = 'coherent_state_' + 'q' + str(q_init).replace(".", "") + 'p_init' + str(p_init).replace(".", "")

##### Save all of the data
data_output_path = './Data/qm_sse_simulation_ind_path_' + init_state_str + \
                   '_t' + str(T).replace(".", "") + '_g' + str(gamma).replace(".", "") + '.mat'

scipy.io.savemat(data_output_path, 
                 mdict={'time_array': time,
                        'init_state': init_state,
                        'x_operator_expectation': results[0],
                        'x_operator_expectation_error': results[1],
                        }, 
                 oned_as='row')
matdata = scipy.io.loadmat(data_output_path)
assert np.all(time == matdata['time_array'])
assert np.all(results[0] == matdata['x_operator_expectation'])
assert np.all(results[1] == matdata['x_operator_expectation_error'])

#### ADDING TEMPERATURE TO THE SYSTEM

# Figure 3.1
gamma = 0.1
T = 0.01

# Initial conditions
percentile = np.random.rand()
p_init = 0.1 #norminv(percentile, 0, np.sqrt(T))
q_init = 1
init_wave = init_coherent_state(dim, p_init, q_init, mass, omega, hbar)

# Defining operators
x_operator = np.sqrt(hbar / (2 * mass * omega)) * (raising_operator(dim) + lowering_operator(dim)) 
p_operator = 1j * np.sqrt(hbar * mass * omega / 2) * (raising_operator(dim) - lowering_operator(dim)) 
lindblad = np.sqrt(2 * mass * kb * T) / hbar * x_operator + 1j / (2 * np.sqrt(2 * mass * kb * T)) * p_operator
h_sys = 1/(2 * mass) * p_operator * p_operator + V2 * x_operator * x_operator * x_operator * x_operator - V1 * x_operator * x_operator
hamiltonian = h_sys + gamma/2 * (x_operator * p_operator + p_operator * x_operator)
time = np.arange(0, no_time_steps * time_step, time_step, dtype = float)
#unseeding code in a cheeky way
np.random.seed(int(time_clock.time()))
results = operator_time_average(init_state, hamiltonian, lindblad, x_operator, hbar, time_step, no_time_steps, 1, "rk")
init_state_str = 'coherent_state_' + 'q' + str(q_init).replace(".", "") + 'p_init' + str(p_init).replace(".", "")

##### Save all of the data
data_output_path = './Data/qm_sse_simulation_ind_path_' + init_state_str + \
                   '_t' + str(T).replace(".", "") + '_g' + str(gamma).replace(".", "") + '.mat'

scipy.io.savemat(data_output_path, 
                 mdict={'time_array': time,
                        'init_state': init_state,
                        'x_operator_expectation': results[0],
                        'x_operator_expectation_error': results[1],
                        }, 
                 oned_as='row')
matdata = scipy.io.loadmat(data_output_path)
assert np.all(time == matdata['time_array'])
assert np.all(results[0] == matdata['x_operator_expectation'])
assert np.all(results[1] == matdata['x_operator_expectation_error'])