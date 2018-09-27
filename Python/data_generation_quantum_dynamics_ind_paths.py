# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 17:52:05 2018

@author: Kamil
"""
import quantum_functions_library as qfl 
import matplotlib.pyplot as plt
import numpy as np
import datetime
import scipy.io

print(datetime.datetime.now())

#### Parameters of model that will not change    
### v1 is the x^2 coefficient
### v2 is the x^4 coefficient
V1 = 1.5
V2 = 0.75
mass = 1
hbar = 1/(8.575*10 **(-34)/ (1.0545718 * 10**(-34)))
kb = 1
omega = 1
time_step = 0.001
no_time_steps = 500000
dim = 80

#### Defining operators that will remain fixed
x_operator = np.sqrt(hbar / (2 * mass * omega)) * (qfl.raising_operator(dim) + qfl.lowering_operator(dim)) 
p_operator = 1j * np.sqrt(hbar * mass * omega / 2) * (qfl.raising_operator(dim) - qfl.lowering_operator(dim)) 
h_sys = 1/(2 * mass) * p_operator * p_operator + V2 * x_operator * x_operator * x_operator * x_operator - V1 * x_operator * x_operator

#### Varying parameters
T = 0.0
gamma = 0.00
# Initial conditions
percentile = np.random.rand()
p_init = 0.1 
q_init = 1
init_state = qfl.init_coherent_state(dim, p_init, q_init, mass, omega, hbar)
init_state_str = 'coherent_state_' + 'q' + str(q_init).replace(".", "") + 'p_init' + str(p_init).replace(".", "")

##### Derived from variables
lindblad = x_operator * np.sqrt(4 * mass * kb * T * gamma) / hbar * x_operator + 1j * np.sqrt(gamma) / (np.sqrt(4 * mass * kb * T)) * p_operator
hamiltonian = h_sys + gamma/2 * (x_operator * p_operator + p_operator * x_operator)

#### Computing the expectation value
results = qfl.two_operator_time_average(init_state, hamiltonian, lindblad, x_operator, p_operator, hbar, time_step, no_time_steps, 1, "rk")

##### Save all of the data
data_output_path = './Data/qm_sse_simulation_ind_path_' + init_state_str + \
                   '_t' + str(T).replace(".", "") + '_g' + str(gamma).replace(".", "") + '.mat'

scipy.io.savemat(data_output_path, 
                 mdict={'time_array': results[0],
                        'x_operator_time_average': results[1],
                        'x_time_standard_error': results[2],
                        'p_operator_time_average': results[3],
                        'p_time_standard_error': results[4],
                        'wave_evol_matrix': results[5]}, 
                 oned_as='row')
matdata = scipy.io.loadmat(data_output_path)
assert np.all(results[0] == matdata['time_array'])
assert np.all(results[1] == matdata['x_operator_time_average'])
assert np.all(results[2] == matdata['x_time_standard_error'])
assert np.all(results[3] == matdata['p_operator_time_average'])
assert np.all(results[4] == matdata['p_time_standard_error'])
assert np.all(results[5] == matdata['wave_evol_matrix'])
print("checkpoint2")
print(datetime.datetime.now())