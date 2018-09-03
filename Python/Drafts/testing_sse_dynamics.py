# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 17:48:19 2018

@author: Kamil
"""
from sse_chiara_2 import *
import matplotlib.pyplot as plt
import timeit
import numpy as np

##### MODEL 1

# Implementing simulation of Eq. 4.1 from Gisin and Percival.
    
# Create initial state as vector (wrong shape, hence the transpose)
init_state = np.matrix([[0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0.0, 0.0]], dtype = complex)
init_state = init_state.T

hamiltonian = 2j * (raising_operator(11) - lowering_operator(11))
lindblad = lowering_operator(11)
gamma = 1.0

# Time step has to be this order of magnitude for Euler scheme
time_step = 0.01
no_time_steps = 1000
time = np.arange(0, no_time_steps * time_step, 0.01, dtype = float)
operator = raising_operator(11) * lowering_operator(11)
results = operator_time_average(init_state, hamiltonian, lindblad, operator, gamma, time_step, no_time_steps, 1, "platen")
plt.errorbar(time, results[0], yerr=results[1]) 

### MODEL 2

# Implementing simulation of Eq. 4.2 from Gisin and Percival

# Create initial state as vector (wrong shape, hence the transpose)
init_state = np.matrix([[1.0, 0, 0, 0]], dtype = complex)
init_state = init_state.T

hamiltonian = 0.1j * (raising_operator(4) - lowering_operator(4))
lindblad = lowering_operator(4) * lowering_operator(4)
gamma = 2.0

## Time step has to be this order of magnitude for Euler scheme
time_step = 0.01
no_time_steps = 80000

wave_evol = simulate_sse_euler(init_state, hamiltonian, lindblad, gamma, time_step, no_time_steps)

# For plot comparison
time = np.empty(no_time_steps, dtype = float)
no_of_photons = np.empty(no_time_steps, dtype = float)
for j in range(wave_evol.shape[1]):
    operator = raising_operator(4) * lowering_operator(4)
    wave_func = np.matrix(wave_evol[:, j], dtype = complex).T
    no_of_photons[j] = expectation(operator, wave_func)
    time[j] = j * time_step

plt.plot(time, no_of_photons)
    


### MODEL 3    
    
#Implementing simulation of Eq. 7.58 from Bruer and Petruccione
    
# Create initial state as vector (wrong shape, hence the transpose)
init_state = np.matrix([[0, 0, 0, 0, 0, 0, 0, 0, 0.0, 1.0, 0.0, 0.0, 0.0]], dtype = complex)
init_state = init_state.T

hamiltonian = 2j * (raising_operator(13) - raising_operator(13))
lindblad = lowering_operator(13)
gamma = 0.8

# Time step has to be this order of magnitude for Euler scheme
time_step = 0.0001
no_time_steps = 10000

wave_evol = simulate_sse_euler(init_state, hamiltonian, lindblad, gamma, time_step, no_time_steps)

# For plot comparison
time = np.empty(no_time_steps, dtype = float)
no_of_photons = np.empty(no_time_steps, dtype = float)
for j in range(wave_evol.shape[1]):
    operator = raising_operator(13) * lowering_operator(13)
    wave_func = np.matrix(wave_evol[:, j], dtype = complex).T
    no_of_photons[j] = expectation(operator, wave_func)
    time[j] = j * time_step * gamma

plt.plot(time, no_of_photons)
    
    
   
### MODEL 4    

# Implementing driven two-level system based on equation 7.60 from Bruer and Petrucione
# "Theory of open quantum systems"

# Create initial state as vector (wrong shape, hence the transpose)
init_state = np.matrix([[0.0, 1.0]], dtype = complex)
init_state = init_state.T

omega = 0.4
gamma = 0.4

sigma_raise = np.matrix([[0.0, 1.0], [0.0, 0.0]], dtype = complex)
sigma_lower = np.matrix([[0.0, 0.0], [1.0, 0.0]], dtype = complex)
hamiltonian = - omega/2 * (sigma_lower + sigma_raise)
lindblad = sigma_lower

# Time step has to be this order of magnitude for Euler scheme
time_step = 0.01
no_time_steps = 5000

# For plot comparison
time = np.linspace(0, 20, 5000)
ro_11 = np.zeros(no_time_steps, dtype = float)
exc_state = np.matrix([[1.0, 0.0]], dtype = complex)
wave_evol = simulate_sse_euler(init_state, hamiltonian, lindblad, gamma, time_step, no_time_steps)
for j in range(wave_evol.shape[1]):
    ro_11[j] = wave_evol[:,j][0] * wave_evol[:,j][0].conj()

plt.figure()
plt.plot(time, ro_11)


# Computing average over a number of simulations of the stochastic Schrodinger
# equation. Plotting both the average and standard error, to later plot it. At
# numba.jit is used to parallelise but whether it is actually quicker not yet
# shown. FIX ME (if possible)
no_of_realisations = 100
    
# Computing the averaged value over 1000 realisations
@numba.jit(parallel=True)
def average(init_state, hamiltonian, lindblad, gamma, time_step, no_time_steps, no_of_realisations):
    results = np.zeros((no_time_steps, no_of_realisations), dtype = float)    
    for j in range(no_of_realisations):
        wave_evol = simulate_sse_platen(init_state, hamiltonian, lindblad, gamma, time_step, no_time_steps)
        for k in range(wave_evol.shape[1]):
            results[k, j] += wave_evol[:,k][0] * wave_evol[:,k][0].conj()
    return(results)
    
results = average(init_state, hamiltonian, lindblad, gamma, time_step, no_time_steps, no_of_realisations)

average = np.mean(results, axis = 1)
standard_error = np.zeros((no_time_steps, no_of_realisations), dtype = float)

for j in range(results.shape[1]):
    standard_error[:, j] = results[:, j] - average
    
standard_error = np.multiply(standard_error, standard_error)
standard_error = np.sum(standard_error, axis = 1)
standard_error = standard_error/(len(standard_error) * (len(standard_error) - 1))
standard_error = np.sqrt(standard_error)
plt.errorbar(time, average, yerr=standard_error)

