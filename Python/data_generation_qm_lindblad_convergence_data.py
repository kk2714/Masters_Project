# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 12:29:44 2018

@author: Kamil

Script for computing the x and p expectation values for solved Liouville equation
for the density matrix. Data required for convergence visualisation.
"""

import quantum_functions_library as qfl 
import numpy as np
import matplotlib.pyplot as plt
import numba
import timeit
from matplotlib import animation
import numpy as np
import scipy.io
import sys, os

### Parameters of model - for my project these will remain constant, but can 
### be adjusted if required to look at other problems.

### v1 is the x^2 coefficient in V(x)
### v2 is the x^4 coefficient in V(x)
V1 = 1.5
V2 = 0.75
mass = 1
hbar = 1/(8.575*10 **(-34)/ (1.0545718 * 10**(-34)))
kb = 1
omega = 1
dim = 80

### Defining operators independent of T and gamma.
x_operator = np.sqrt(hbar / (2 * mass * omega)) * (qfl.raising_operator(dim) + qfl.lowering_operator(dim)) 
p_operator = 1j * np.sqrt(hbar * mass * omega / 2) * (qfl.raising_operator(dim) - qfl.lowering_operator(dim)) 
h_sys = 1/(2 * mass) * p_operator * p_operator + V2 * x_operator * x_operator * x_operator * x_operator - V1 * x_operator * x_operator

##### Have checkpoints to debug code. Also allow to see which calculations have already been carried out
print("checkpoint1")

#### Load all of the necessary operators
try:
    data_path = './Data/qm_tools_dim_' + str(dim).replace(".", "") + '_mass_' + str(mass).replace(".", "") + '_omega_' + \
                str(omega).replace(".", "") + '.mat'
    matdata = scipy.io.loadmat(data_path)
    project_A = np.matrix(matdata['project_A'])
    x_grid_projectors = matdata['x_grid_projectors']
    for j in range(len(x_grid_projectors)):
        x_grid_projectors[j] = np.matrix(x_grid_projectors[j])
    x_grid_midpoints = matdata['x_grid_midpoints']
    xmin = matdata['x_limits'][0][0]
    xmax = matdata['x_limits'][0][1]
except Exception as e:
    raise('Data has to be generated first')

### Defining constants and operators that will be adjusted in project
T = 0.25
gamma = 0.01/2
lindblad = np.sqrt(4 * mass * kb * T * gamma) / hbar * x_operator + 1j * np.sqrt(gamma) / (np.sqrt(4 * mass * kb * T)) * p_operator
hamiltonian = h_sys + gamma/2 * (x_operator * p_operator + p_operator * x_operator)

### Start from this wavefunction
p_init = 0.1 
q_init = 1
init_state = qfl.init_coherent_state(dim, p_init, q_init, mass, omega, hbar)
init_state_str = 'coherent_state_' + 'q' + str(q_init).replace(".", "") + 'p_init' + str(p_init).replace(".", "")

init_rho = init_state * init_state.getH()
init_rho = init_rho/qfl.trace_matrix(init_rho)

#### RESULTS START HERE

#### Getting the time evolution of rho and prob density

### For most of graphs
short_time_step = 0.001
short_no_steps = 5000

rho_operator_short_time = qfl.rho_lindblad_superoperator(init_rho, hamiltonian, lindblad, hbar, short_time_step, short_no_steps)

print("checkpoint5")

short_time = np.arange(0, short_no_steps * short_time_step, short_time_step)

@numba.jit
def exp_x_time(rho_operator, time):
    x_operator_average = np.empty(len(time), dtype = float)
    p_operator_average = np.empty(len(time), dtype = float)    
    for j in range(len(time)):
        rho = np.matrix(rho_operator[:, :, j])
        x_operator_average[j] = qfl.trace_matrix(rho * x_operator)
        p_operator_average[j] = qfl.trace_matrix(rho * p_operator)
    return x_operator_average, p_operator_average

x_operator_average, p_operator_average = exp_x_time(rho_operator_short_time, short_time)
plt.plot(x_operator_average)
##### Save all of the data
data_output_path = './Data/qm_lindblad_simulation_conv_data' + init_state_str + \
                   '_t' + str(T).replace(".", "") + '_g' + str(gamma).replace(".", "") + '.mat'

scipy.io.savemat(data_output_path, 
                 mdict={'lindblad': lindblad,
                        'hamiltonian': hamiltonian,
                        'init_rho': init_rho,
                        'rho_operator_short_time': rho_operator_short_time,
                        'x_operator_average': x_operator_average,
                        'p_operator_average': p_operator_average,
                        }, 
                 oned_as='row')
matdata = scipy.io.loadmat(data_output_path)
assert np.all(rho_operator_short_time == matdata['rho_operator_short_time'])
assert np.all(x_operator_average == matdata['x_operator_average'])
assert np.all(p_operator_average == matdata['p_operator_average'])
print("checkpoint8")