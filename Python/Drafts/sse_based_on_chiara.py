# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 23:46:30 2018

@author: Kamil
"""

import numpy as np
import matplotlib.pyplot as plt
import numba
from scipy.stats import norm
from math import factorial

# Simulating quantum system using stochastic Schrodinger equation, numba compiler
# and different numerical methods

# Define inverse of Gaussian distribution
def norminv(percentile, mean, stddev):
    return norm.ppf(percentile, loc = mean, scale = stddev)

# Define lowering operator
def lowering_operator(dim):
    A = np.matrix([[0 for x in range(dim)] for y in range(dim)], dtype = complex)
    for i in range(dim - 1):
        A[i, i + 1] = np.sqrt(i + 1)
    A[dim - 1, dim - 1] = 1.0
    return(A)
        
# Define raising operator
def raising_operator(dim):
    A = np.matrix([[0 for x in range(dim)] for y in range(dim)], dtype = complex)
    for i in range(1, dim):
        A[i, i - 1] = np.sqrt(i)
    return(A)
    
# Define a function to compute the expectation of an operator
@numba.jit
def expectation(operator, wavefunction):
    wave_dagger = wavefunction.getH()
    exp_num_matrix = wave_dagger * operator * wavefunction
    exp_den_matrix = wave_dagger * wavefunction
    expectation = exp_num_matrix[0,0]/exp_den_matrix[0,0]
    return expectation

# Implementing different schemes to simulate stochastic Schrodinger equation. Based upon
# equation C.8 from appendix C of Chiara Liverani's thesis "novel approach to quantum
# transition rate theory using open quantum dynamics".

@numba.jit
def drift_coeff(wavefunction, hamiltonian, lindblad, gamma):
    'All inputs have to be as np.matrix format. Returns a np.matrix'
    exp_val = expectation(lindblad, wavefunction)
    exp_val_H = expectation(lindblad.getH(), wavefunction)
    hbar = 1/(8.575*10 **(-34)/ (1.0545718 * 10**(-34)))
    drift = -1j/hbar * hamiltonian * wavefunction + gamma * 2 * exp_val_H * lindblad * wavefunction - \
            gamma * lindblad.getH() * lindblad * wavefunction - gamma * exp_val_H * exp_val * wavefunction
    return(drift)

@numba.jit
def diffusion_term(wavefunction, hamiltonian, lindblad, gamma):
    'All inputs have to be as np.matrix format. Returns a np.matrix'
    exp_val = expectation(lindblad, wavefunction)
    diffusion = np.sqrt(gamma) * lindblad * wavefunction - np.sqrt(gamma) * exp_val * wavefunction
    return(diffusion)


# Implementing Euler scheme. Section 7.2.2 of "Theory of Open Quantum Systems". 
# Bruer and Petruccione

@numba.jit
def simulate_sse_euler(init_state, hamiltonian, lindblad, gamma, time_step, no_time_steps):
    n = init_state.shape[0]
    wave_evol = np.empty((n, no_time_steps), dtype = complex)
    wave_evol[:, 0] = np.asarray(init_state).reshape(-1)
    k = 1
    for i in range(0, no_time_steps - 1):
        wave_prev = np.matrix(wave_evol[:, k-1], dtype = complex).T
        drift = drift_coeff(wave_prev, hamiltonian, lindblad, gamma)
        diffusion = diffusion_term(wave_prev, hamiltonian, lindblad, gamma)
        wave_after =  wave_prev + drift * time_step + diffusion * np.sqrt(time_step) * (np.random.normal(0,1) + 1j * np.random.normal(0,1))
        wave_after = np.asarray(wave_after).reshape(-1)
        wave_evol[:, k] = wave_after/np.linalg.norm(wave_after)
        k += 1
    return(wave_evol)

# Implementing Heun scheme. Section 7.2.3 of "Theory of Open Quantum Systems". 
# Bruer and Petruccione
    
@numba.jit
def simulate_sse_heun(init_state, hamiltonian, lindblad, gamma, time_step, no_time_steps):
    n = init_state.shape[0]
    wave_evol = np.empty((n, no_time_steps), dtype = complex)
    wave_evol[:, 0] = np.asarray(init_state).reshape(-1)
    k = 1
    for i in range(0, no_time_steps - 1):
        # Drift coefficient computed twice and averaged
        wave_step_before = np.matrix(wave_evol[:, k-1], dtype = complex).T
        diffusion = diffusion_term(wave_step_before, hamiltonian, lindblad, gamma)
        
        # Computing Wiener increment
        wiener = np.sqrt(time_step) * (np.random.normal(0,1) + 1j * np.random.normal(0,1))
        
        # Computing additional wavefunction and drifts
        drift_1 = drift_coeff(wave_step_before, hamiltonian, lindblad, gamma)
        wave_step_interm = wave_step_before + drift_1 * time_step + diffusion * wiener
        drift_2 = drift_coeff(wave_step_interm, hamiltonian, lindblad, gamma)
        
        # Final wavefunction
        wave_step_after =  wave_step_before + 0.5 * (drift_1 + drift_2) * time_step + diffusion * wiener
        wave_step_after = np.asarray(wave_step_after).reshape(-1)
        wave_evol[:, k] = wave_step_after/np.linalg.norm(wave_step_after)
        k += 1
    return(wave_evol)

# Implementing Runge-Kutta scheme. Section 7.2.4 of "Theory of Open Quantum Systems". 
# Bruer and Petruccione
    
@numba.jit
def simulate_sse_rk(init_state, hamiltonian, lindblad, gamma, time_step, no_time_steps):
    n = init_state.shape[0]
    wave_evol = np.empty((n, no_time_steps), dtype = complex)
    wave_evol[:, 0] = np.asarray(init_state).reshape(-1)
    k = 1
    for i in range(0, no_time_steps - 1):
        # Drift coefficient computed four times and weighted averaged
        wave_step_before = np.matrix(wave_evol[:, k-1], dtype = complex).T
        diffusion = diffusion_term(wave_step_before, hamiltonian, lindblad, gamma)
        
        # Computing intermediate wavefunction and drifts
        drift_1 = drift_coeff(wave_step_before, hamiltonian, lindblad, gamma)
        wave_interm_1 = wave_step_before + 0.5 * time_step * drift_1
        
        drift_2 = drift_coeff(wave_interm_1, hamiltonian, lindblad, gamma)
        wave_interm2 = wave_step_before + 0.5 * time_step * drift_2
        
        drift_3 = drift_coeff(wave_interm2, hamiltonian, lindblad, gamma)
        wave_interm3 = wave_step_before + time_step * drift_3
        
        drift_4 = drift_coeff(wave_interm3, hamiltonian, lindblad, gamma)
        
        drift_overall = 1/6 * (drift_1 + 2 * drift_2 + 2 * drift_3 + drift_4)
        
        # Final wavefunction
        wave_step_after =  wave_step_before + drift_overall * time_step + diffusion * np.sqrt(time_step) * (np.random.normal(0,1) + 1j * np.random.normal(0,1))
        wave_step_after = np.asarray(wave_step_after).reshape(-1)
        wave_evol[:, k] = wave_step_after/np.linalg.norm(wave_step_after)
        k += 1
    return(wave_evol)

# Implementing Platen scheme. Section 7.2.5 of "Theory of Open Quantum Systems". 
# Bruer and Petruccione
    
@numba.jit
def simulate_sse_platen(init_state, hamiltonian, lindblad, gamma, time_step, no_time_steps):
    n = init_state.shape[0]
    wave_evol = np.empty((n, no_time_steps), dtype = complex)
    wave_evol[:, 0] = np.asarray(init_state).reshape(-1)
    k = 1
    for i in range(0, no_time_steps - 1):
        # Computing drift and diffusion terms based on previous value
        wave_step_before = np.matrix(wave_evol[:, k-1], dtype = complex).T
        diffusion = diffusion_term(wave_step_before, hamiltonian, lindblad, gamma)
        drift = drift_coeff(wave_step_before, hamiltonian, lindblad, gamma)
        
        # Computing Wiener increment
        wiener = np.sqrt(time_step) * (np.random.normal(0,1) + 1j * np.random.normal(0,1))
        
        # Computing intermediate wavefunction
        wave_interm_hat = wave_step_before + drift * time_step + diffusion * wiener
        wave_interm_pos = wave_step_before + drift * time_step + diffusion * np.sqrt(time_step)
        wave_interm_neg = wave_step_before + drift * time_step - diffusion * np.sqrt(time_step)
        
        # Computing the remaining necessary drift terms
        drift_hat = drift_coeff(wave_interm_hat, hamiltonian, lindblad, gamma)
        
        diffusion_pos = diffusion_term(wave_interm_pos, hamiltonian, lindblad, gamma)
        diffusion_neg = diffusion_term(wave_interm_neg, hamiltonian, lindblad, gamma)
        
        # For computing wavefunction on previous value
        drift_overall = 0.5 * (drift + drift_hat)
        diffusion_overall = 0.25 * (diffusion_pos + diffusion_neg + 2 * diffusion)
        diffusion_add = 0.25 * (diffusion_pos - diffusion_neg)
        
        # Final wavefunction
        wave_step_after =  wave_step_before + drift_overall * time_step + diffusion_overall * wiener + \
                           diffusion_add * 1/np.sqrt(time_step) * (wiener ** 2 - time_step) 
        
        wave_step_after = np.asarray(wave_step_after).reshape(-1)
        wave_evol[:, k] = wave_step_after/np.linalg.norm(wave_step_after)
        k += 1
    return(wave_evol)

### MODEL 1

# Implementing simulation of Eq. 4.1 from Gisin and Percival.
    
# Create initial state as vector (wrong shape, hence the transpose)
init_state = np.matrix([[0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0.0, 0.0]], dtype = complex)
init_state = init_state.T

hamiltonian = 2j * (raising_operator(11) - lowering_operator(11))
lindblad = lowering_operator(11)
gamma = 1.0

# Time step has to be this order of magnitude for Euler scheme
time_step = 0.01
no_time_steps = 10000

# Seeding the code
np.random.seed(9001)
wave_evol = simulate_sse_euler(init_state, hamiltonian, lindblad, gamma, time_step, no_time_steps)

# For plot comparison
time = np.empty(no_time_steps, dtype = float)
no_of_photons = np.empty(no_time_steps, dtype = float)
for j in range(wave_evol.shape[1]):
    operator = raising_operator(11) * lowering_operator(11)
    wave_func = np.matrix(wave_evol[:, j], dtype = complex).T
    no_of_photons[j] = expectation(operator, wave_func)
    time[j] = j * time_step

plt.plot(time, no_of_photons)
plt.ylabel('No. of particles')
plt.xlabel('time (s)')



#### MODEL 2

## Implementing simulation of Eq. 4.2 from Gisin and Percival
#
## Create initial state as vector (wrong shape, hence the transpose)
#init_state = np.matrix([[1.0, 0, 0, 0]], dtype = complex)
#init_state = init_state.T
#
#hamiltonian = 0.1j * (raising_operator(4) - lowering_operator(4))
#lindblad = lowering_operator(4) * lowering_operator(4)
#gamma = 1.0
#
### Time step has to be this order of magnitude for Euler scheme
#time_step = 0.01
#no_time_steps = 80000
#
#wave_evol = simulate_sse_euler(init_state, hamiltonian, lindblad, gamma, time_step, no_time_steps)
#
## For plot comparison
#time = np.empty(no_time_steps, dtype = float)
#no_of_photons = np.empty(no_time_steps, dtype = float)
#for j in range(wave_evol.shape[1]):
#    operator = raising_operator(4) * lowering_operator(4)
#    wave_func = np.matrix(wave_evol[:, j], dtype = complex).T
#    no_of_photons[j] = expectation(operator, wave_func)
#    time[j] = j * time_step
#
#plt.plot(time, no_of_photons)
    


#### MODEL 3    
    
##Implementing simulation of Eq. 7.58 from Bruer and Petruccione
#    
## Create initial state as vector (wrong shape, hence the transpose)
#init_state = np.matrix([[0, 0, 0, 0, 0, 0, 0, 0, 0.0, 1.0, 0.0, 0.0, 0.0]], dtype = complex)
#init_state = init_state.T
#
#hamiltonian = 2j * (raising_operator(13) - raising_operator(13))
#lindblad = lowering_operator(13)
#gamma = 0.4
#
## Time step has to be this order of magnitude for Euler scheme
#time_step = 0.0001
#no_time_steps = 10000
#
#wave_evol = simulate_sse_euler(init_state, hamiltonian, lindblad, gamma, time_step, no_time_steps)
#
## For plot comparison
#time = np.empty(no_time_steps, dtype = float)
#no_of_photons = np.empty(no_time_steps, dtype = float)
#for j in range(wave_evol.shape[1]):
#    operator = raising_operator(13) * lowering_operator(13)
#    wave_func = np.matrix(wave_evol[:, j], dtype = complex).T
#    no_of_photons[j] = expectation(operator, wave_func)
#    time[j] = j * time_step * gamma
#
#plt.plot(time, no_of_photons)
    
    
   
##### MODEL 4    
#
## Implementing driven two-level system based on equation 7.60 from Bruer and Petrucione
## "Theory of open quantum systems"
#
## Create initial state as vector (wrong shape, hence the transpose)
#init_state = np.matrix([[0.0, 1.0]], dtype = complex)
#init_state = init_state.T
#
#omega = 0.4
#gamma = 0.2
#
#sigma_raise = np.matrix([[0.0, 1.0], [0.0, 0.0]], dtype = complex)
#sigma_lower = np.matrix([[0.0, 0.0], [1.0, 0.0]], dtype = complex)
#hamiltonian = - omega/2 * (sigma_lower + sigma_raise)
#lindblad = sigma_lower

## Time step has to be this order of magnitude for Euler scheme
#time_step = 0.01
#no_time_steps = 5000

## For plot comparison
#time = np.linspace(0, 20, 5000)
#ro_11 = np.zeros(no_time_steps, dtype = float)
#exc_state = np.matrix([[1.0, 0.0]], dtype = complex)
#wave_evol = simulate_sse_platen(init_state, hamiltonian, lindblad, gamma, time_step, no_time_steps)
#for j in range(wave_evol.shape[1]):
#    ro_11[j] = wave_evol[:,j][0] * wave_evol[:,j][0].conj()
#
#plt.plot(time, ro_11)
    

## Computing average over a number of simulations of the stochastic Schrodinger
## equation. Plotting both the average and standard error, to later plot it. At
## numba.jit is used to parallelise but whether it is actually quicker not yet
## shown. FIX ME (if possible)
#no_of_realisations = 100
#    
## Computing the averaged value over 1000 realisations
#@numba.jit(parallel=True)
#def average(init_state, hamiltonian, lindblad, gamma, time_step, no_time_steps, no_of_realisations):
#    results = np.zeros((no_time_steps, no_of_realisations), dtype = float)    
#    for j in range(no_of_realisations):
#        wave_evol = simulate_sse_heun(init_state, hamiltonian, lindblad, gamma, time_step, no_time_steps)
#        for k in range(wave_evol.shape[1]):
#            results[k, j] += wave_evol[:,k][0] * wave_evol[:,k][0].conj()
#    return(results)
#    
#results = average(init_state, hamiltonian, lindblad, gamma, time_step, no_time_steps, no_of_realisations)
#
#average = np.mean(results, axis = 1)
#standard_error = np.zeros((no_time_steps, no_of_realisations), dtype = float)
#
#for j in range(results.shape[1]):
#    standard_error[:, j] = results[:, j] - average
#    
#standard_error = np.multiply(standard_error, standard_error)
#standard_error = np.sum(standard_error, axis = 1)
#standard_error = standard_error/(len(standard_error) * (len(standard_error) - 1))
#standard_error = np.sqrt(standard_error)
#plt.errorbar(time, average, yerr=standard_error)
#plt.ylabel('Prob of being in state 1')
#plt.xlabel('time (s)')

## Recreating Chiara's stochastic simulations
    
def init_coherent_state(dim, p_init, q_init, mass, omega):
    hbar = 1/(8.575*10 **(-34)/ (1.0545718 * 10**(-34)))
    z = np.sqrt(mass * omega / (2 * hbar)) * q_init + 1j * 1/np.sqrt(2 * mass * omega * hbar) * p_init
    wavefunction = np.zeros((1, dim), dtype = complex)
    for j in range(dim):
        wavefunction[:, j] = np.exp(-0.5 * z * z.conj()) * z ** j / np.sqrt(float(factorial(j)))
    wavefunction = np.matrix(wavefunction, dtype = complex).T
    
    # Renormalise to account for lack of higher order states
    wavefunction = wavefunction/np.linalg.norm(wavefunction)
    return(wavefunction)

