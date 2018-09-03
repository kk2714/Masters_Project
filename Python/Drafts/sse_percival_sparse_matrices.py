# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 23:46:30 2018

@author: Kamil
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
import matplotlib.pyplot as plt

# to convert to normal matrix to print sp.sparse.csr_matrix.todense

# Simulating quantum system using stochastic Schrodinger equation and sparse matrices

# Define lowering operator
def lowering_operator(dim):
    A = sp.lil_matrix((dim, dim))
    for i in range(dim - 1):
        A[np.ix_([i], [i + 1])] = np.sqrt(i + 1)
    A[np.ix_([dim - 1], [dim - 1])] = 1.0
    return(A)
        
# Define raising operator
def raising_operator(dim):
    A = sp.lil_matrix((dim, dim))
    for i in range(1, dim):
        A[np.ix_([i], [i - 1])] = np.sqrt(i)
    return(A)
    
# Define a function to compute the expectation of an operator
def expectation(operator, wavefunction):
    wave_dagger = wavefunction.getH()
    exp_num_matrix = wave_dagger * operator * wavefunction
    exp_den_matrix = wave_dagger * wavefunction
    expectation = exp_num_matrix[0,0]/exp_den_matrix[0,0]
    return expectation

# Implementing Euler scheme to simulate stochastic Schrodinger equation. Based upon
# equation 3.1 from N Gisin and IC Percival "The quantum-state diffusion model 
# applied to open systems" 1992 J. Phys. A: Math. Gen. 25 5677.
def simulate_sse(init_state, ham, lin, time_step, no_time_steps):
    n = init_state.shape[0]
    wave_evol = sp.lil_matrix((n, no_time_steps), dtype = complex)
    wave_evol[:, 0] = init_state
    hbar = 1 #1.05 * 10 ** (-34)
    k = 1
    d_1 = - (1j/hbar) * ham - lin.getH() * lin
    for i in range(0, no_time_steps - 1): 
        wave_evol[:, k] = wave_evol[:, k-1]  + (d_1 + 2 * expectation(lin.getH(), wave_evol[:, k-1]) * lin) * wave_evol[:, k-1] * time_step + lin * wave_evol[:, k-1] * np.sqrt(time_step) * np.random.normal(0,1)
        wave_evol[:, k] = wave_evol[:, k]/sp.linalg.norm(wave_evol[:, k])
        k += 1
    return(wave_evol)

# Implementing simulation of Eq. 4.1 from Gisin and Percival
    
## Create initial state as vector (wrong shape, hence the transpose)
#init_state = np.matrix([[0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0.0]], dtype = complex)
#init_state = init_state.T
#
#hamiltonian = 2j * (raising_operator(10) - lowering_operator(10))
#lindblad = lowering_operator(10)

## Time step has to be this order of magnitude for Euler scheme
#time_step = 0.001
#no_time_steps = 10000

#wave_evol = simulate_sse(init_state, hamiltonian, lindblad, time_step, no_time_steps)

## For plot comparison
#time = np.empty(no_time_steps, dtype = float)
#no_of_photons = np.empty(no_time_steps, dtype = float)
#for j in range(wave_evol.shape[1]):
#    operator = raising_operator(10) * lowering_operator(10)
#    no_of_photons[j] = expectation(operator, wave_evol[:, j])
#    time[j] = j * time_step
#
#plt.plot(time, no_of_photons)



# Implementing simulation of Eq. 4.2 from Gisin and Percival

# Create initial state as vector (wrong shape, hence the transpose)
init_state = np.matrix([[1.0, 0, 0, 0]], dtype = complex)
init_state = init_state.T

hamiltonian = 0.1j * (raising_operator(4) - lowering_operator(4))
lindblad = lowering_operator(4) * lowering_operator(4)

# Time step has to be this order of magnitude for Euler scheme
time_step = 0.01
no_time_steps = 80000

wave_evol = simulate_sse(init_state, hamiltonian, lindblad, time_step, no_time_steps)

# For plot comparison
time = np.empty(no_time_steps, dtype = float)
no_of_photons = np.empty(no_time_steps, dtype = float)
for j in range(wave_evol.shape[1]):
    operator = raising_operator(4) * lowering_operator(4)
    no_of_photons[j] = expectation(operator, wave_evol[:, j])
    time[j] = j * time_step

plt.plot(time, no_of_photons)