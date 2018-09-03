# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 23:46:30 2018

@author: Kamil
"""

import numpy as np
import matplotlib.pyplot as plt
import numba
from random import gauss
from math import factorial
from scipy.stats import norm
import time as time_clock

# Simulating quantum system using stochastic Schrodinger equation, numba compiler
# and different numerical methods

# Define inverse of Gaussian distribution
def norminv(percentile, mean, stddev):
    return norm.ppf(percentile, loc = mean, scale = stddev)

# Define lowering operator
def lowering_operator(dim):
    '''Arguments:
        dim - as integer. Dimension of NxN matrix
      Returns:
        lowering operator (NxN matrix) for simple harmonic oscillator basis'''
    A = np.matrix([[0 for x in range(dim)] for y in range(dim)], dtype = complex)
    for i in range(dim - 1):
        A[i, i + 1] = np.sqrt(i + 1)
    A[dim - 1, dim - 1] = 1.0
    return(A)
        
# Define raising operator
def raising_operator(dim):
    '''Arguments:
        dim - as integer. Dimension of NxN matrix
      Returns:
        raising operator (NxN matrix) for simple harmonic oscillator basis'''
    A = np.matrix([[0 for x in range(dim)] for y in range(dim)], dtype = complex)
    for i in range(1, dim):
        A[i, i - 1] = np.sqrt(i)
    return(A)

# Defining initial coherent state
def init_coherent_state(dim, p_init, q_init, mass, omega):
    '''Arguments:
        dim - as integer. Dimension of N matrix
        p_init - as float. Initial momentum of particle
        q_init - as float. Initial position of particle
        mass - as float. Mass of particle
        omega - as float natural frequency of the simple harmonic oscillator basis.
      Returns:
        A coherent state (N matrix) for simple harmonic oscillator basis'''
    hbar = 1/(8.575*10 **(-34)/ (1.0545718 * 10**(-34)))
    z = np.sqrt(mass * omega / (2 * hbar)) * q_init + 1j * 1/np.sqrt(2 * mass * omega * hbar) * p_init
    wavefunction = np.zeros((1, dim), dtype = complex)
    for j in range(dim):
        wavefunction[:, j] = np.exp(-0.5 * z * z.conj()) * z ** j / np.sqrt(float(factorial(j)))
    wavefunction = np.matrix(wavefunction, dtype = complex).T
    
    # Renormalise to account for lack of higher order states
    wavefunction = wavefunction/np.linalg.norm(wavefunction)
    return(wavefunction)
    
# Define a function to compute the expectation of an operator
@numba.jit
def expectation(operator, wavefunction):
    '''Arguments:
        operator - as np.matrix
        wavefunction - as np.matrix
      Returns:
        Expectation as a float'''
    wave_dagger = wavefunction.getH()
    exp_num_matrix = wave_dagger * operator * wavefunction
    exp_den_matrix = wave_dagger * wavefunction
    expectation = exp_num_matrix[0,0]/exp_den_matrix[0,0]
    return expectation

# Define a function to compute an operator time-average with standard error
@numba.jit(parallel=True)
def operator_time_average(init_state, hamiltonian, lindblad, operator, gamma, time_step, no_time_steps, no_of_realisations, method):
    '''Arguments:
        init_state - N matrix defining initial state. Initial state will be propagated in time
        hamiltonian - NxN matrix defining the Hamiltonian of the system for the Lindblad equation
        lindblad - NxN matrix defining the Lindblad operator for Lindblad equation.
        operator - NxN matrix. Operator, whos time average we are interested in.
        gamma - as float. "Friction" coefficient
        time_step - as float. The time step used in the numerical scheme.
        no_time_steps - as integer. The amount of time steps to propagate through.
        no_of_realisations - as float. No of realisations to average over.
        method - as string. Valid inputs include: "euler", "rk", "heun", "platen". 
      Returns:
        The average over realisation for the time-dependent operator. Additionally returns the standard
        error on the result as a matrix'''
    results = np.zeros((no_time_steps, no_of_realisations), dtype = float)
    if method not in ("euler", "rk", "heun", "platen"):
        raise ValueError("Unknown numerical scheme. Please read function specification.")    
    # Euler method
    if method == "euler":
        for j in range(no_of_realisations):
            wave_evol = simulate_sse_euler(init_state, hamiltonian, lindblad, gamma, time_step, no_time_steps)
            for k in range(wave_evol.shape[1]):
                wave_func = np.matrix(wave_evol[:, k], dtype = complex).T
                results[k, j] += expectation(operator, wave_func)
    # Heun method
    if method == "heun":
        for j in range(no_of_realisations):
            wave_evol = simulate_sse_heun(init_state, hamiltonian, lindblad, gamma, time_step, no_time_steps)
            for k in range(wave_evol.shape[1]):
                wave_func = np.matrix(wave_evol[:, k], dtype = complex).T
                results[k, j] += expectation(operator, wave_func)
    # Runge-Kutta method
    if method == "rk":
        for j in range(no_of_realisations):
            wave_evol = simulate_sse_rk(init_state, hamiltonian, lindblad, gamma, time_step, no_time_steps)
            for k in range(wave_evol.shape[1]):
                wave_func = np.matrix(wave_evol[:, k], dtype = complex).T
                results[k, j] += expectation(operator, wave_func)
    # Platen method
    if method == "platen":
        for j in range(no_of_realisations):
            wave_evol = simulate_sse_platen(init_state, hamiltonian, lindblad, gamma, time_step, no_time_steps)
            for k in range(wave_evol.shape[1]):
                wave_func = np.matrix(wave_evol[:, k], dtype = complex).T
                results[k, j] += expectation(operator, wave_func)
    average = np.mean(results, axis = 1)
    
    # Computing the standard error on expectation value
    if no_of_realisations != 1 and no_of_realisations > 0:
        standard_error = np.zeros((no_time_steps, no_of_realisations), dtype = float)
        for j in range(results.shape[1]):
            standard_error[:, j] = results[:, j] - average
            standard_error = np.multiply(standard_error, standard_error)
            standard_error = np.sum(standard_error, axis = 1)
            standard_error = standard_error/(len(standard_error) * (len(standard_error) - 1))
            standard_error = np.sqrt(standard_error)
    else:
        standard_error = np.zeros(no_time_steps, dtype = float)
    return(average, standard_error)

# Implementing different schemes to simulate stochastic Schrodinger equation. Based upon
# equation 7.27, 7.28 and 7.29 from "The theory of open quantum systems" by Bruer 
# Petruccione.

@numba.jit
def drift_coeff(wavefunction, hamiltonian, lindblad, gamma):
    '''Arguments:
        wavefunction - N matrix defining the current state. 
        hamiltonian - NxN matrix defining the Hamiltonian of the system for the Lindblad equation
        lindblad - NxN matrix defining the Lindblad operator for Lindblad equation.
        gamma - as float. "Friction" coefficient
      Returns:
        The drift term needed for stochastic Schrodinger equation in numerical schemes.'''
    exp_val = expectation(lindblad, wavefunction)
    exp_val_H = expectation(lindblad.getH(), wavefunction)
    hbar = 1/(8.575*10 **(-34)/ (1.0545718 * 10**(-34)))
    drift = -1j/hbar * hamiltonian * wavefunction + gamma * 2 * exp_val_H * lindblad * wavefunction - \
            gamma * lindblad.getH() * lindblad * wavefunction - gamma * exp_val_H * exp_val * wavefunction
    return(drift)

@numba.jit
def diffusion_term(wavefunction, hamiltonian, lindblad, gamma):
    '''Arguments:
        wavefunction - N matrix defining the current state. 
        hamiltonian - NxN matrix defining the Hamiltonian of the system for the Lindblad equation
        lindblad - NxN matrix defining the Lindblad operator for Lindblad equation.
        gamma - as float. "Friction" coefficient
      Returns:
        The diffusion term needed for stochastic Schrodinger equation in numerical schemes.'''
    exp_val = expectation(lindblad, wavefunction)
    diffusion = np.sqrt(gamma) * lindblad * wavefunction - np.sqrt(gamma) * exp_val * wavefunction
    return(diffusion)

# Implementing Euler scheme. Section 7.2.2 of "Theory of Open Quantum Systems". 
# Bruer and Petruccione

@numba.jit
def simulate_sse_euler(init_state, hamiltonian, lindblad, gamma, time_step, no_time_steps):
    '''Arguments:
        init_state - N matrix defining initial state. Initial state will be propagated in time
        hamiltonian - NxN matrix defining the Hamiltonian of the system for the Lindblad equation
        lindblad - NxN matrix defining the Lindblad operator for Lindblad equation.
        gamma - as float. "Friction" coefficient
        time_step - as float. The time step used in the numerical scheme.
        no_time_steps - as integer. The amount of time steps to propagate through.
      Returns:
        The wavefunction at different times propagated according to the Euler numerical scheme.'''
    n = init_state.shape[0]
    wave_evol = np.empty((n, no_time_steps), dtype = complex)
    wave_evol[:, 0] = np.asarray(init_state).reshape(-1)
    k = 1
    for i in range(0, no_time_steps - 1):
        wave_prev = np.matrix(wave_evol[:, k-1], dtype = complex).T
        drift = drift_coeff(wave_prev, hamiltonian, lindblad, gamma)
        diffusion = diffusion_term(wave_prev, hamiltonian, lindblad, gamma)
        wave_after =  wave_prev + drift * time_step + diffusion * np.sqrt(time_step) * np.random.normal(0,1)
        wave_after = np.asarray(wave_after).reshape(-1)
        wave_evol[:, k] = wave_after/np.linalg.norm(wave_after)
        k += 1
    return(wave_evol)

# Implementing Heun scheme. Section 7.2.3 of "Theory of Open Quantum Systems". 
# Bruer and Petruccione
    
@numba.jit
def simulate_sse_heun(init_state, hamiltonian, lindblad, gamma, time_step, no_time_steps):
    '''Arguments:
        init_state - N matrix defining initial state. Initial state will be propagated in time
        hamiltonian - NxN matrix defining the Hamiltonian of the system for the Lindblad equation
        lindblad - NxN matrix defining the Lindblad operator for Lindblad equation.
        gamma - as float. "Friction" coefficient
        time_step - as float. The time step used in the numerical scheme.
        no_time_steps - as integer. The amount of time steps to propagate through.
      Returns:
        The wavefunction at different times propagated according to the Heun numerical scheme.'''
    n = init_state.shape[0]
    wave_evol = np.empty((n, no_time_steps), dtype = complex)
    wave_evol[:, 0] = np.asarray(init_state).reshape(-1)
    k = 1
    for i in range(0, no_time_steps - 1):
        # Drift coefficient computed twice and averaged
        wave_step_before = np.matrix(wave_evol[:, k-1], dtype = complex).T
        diffusion = diffusion_term(wave_step_before, hamiltonian, lindblad, gamma)
        
        # Computing Wiener increment
        wiener = np.sqrt(time_step) * np.random.normal(0,1)
        
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
    '''Arguments:
        init_state - N matrix defining initial state. Initial state will be propagated in time
        hamiltonian - NxN matrix defining the Hamiltonian of the system for the Lindblad equation
        lindblad - NxN matrix defining the Lindblad operator for Lindblad equation.
        gamma - as float. "Friction" coefficient
        time_step - as float. The time step used in the numerical scheme.
        no_time_steps - as integer. The amount of time steps to propagate through.
      Returns:
        The wavefunction at different times propagated according to the Runge-Kutta numerical scheme.'''
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
        wave_step_after =  wave_step_before + drift_overall * time_step + diffusion * np.sqrt(time_step) * np.random.normal(0,1)
        wave_step_after = np.asarray(wave_step_after).reshape(-1)
        wave_evol[:, k] = wave_step_after/np.linalg.norm(wave_step_after)
        k += 1
    return(wave_evol)

# Implementing Platen scheme. Section 7.2.5 of "Theory of Open Quantum Systems". 
# Bruer and Petruccione
    
@numba.jit
def simulate_sse_platen(init_state, hamiltonian, lindblad, gamma, time_step, no_time_steps):
    '''Arguments:
        init_state - N matrix defining initial state. Initial state will be propagated in time
        hamiltonian - NxN matrix defining the Hamiltonian of the system for the Lindblad equation
        lindblad - NxN matrix defining the Lindblad operator for Lindblad equation.
        gamma - as float. "Friction" coefficient
        time_step - as float. The time step used in the numerical scheme.
        no_time_steps - as integer. The amount of time steps to propagate through.
      Returns:
        The wavefunction at different times propagated according to the Platen numerical scheme.'''
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
        wiener = np.sqrt(time_step) * np.random.normal(0,1)
        
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

@numba.jit
def convergence_function(V1, V2, gamma, T, mass, omega, time_step):
    ''' The Hamiltonian is of the form: 0.5 p^2/m - v1 q^2 + v2 q^4. The initial state is a coherent
    state dependepent on the temperature, T. This function computes the optimal dimension for modelling the 
    open quantum dynamics of the system based on the simple harmonic oscillator basis. 
      
    Arguments:
        v1, v2 - floats. Used to describe the quartic energy potential. 
        gamma - as float. "Friction" coefficient in the open quantum system.
        T - temperature of the system
        mass - mass of the particle.
        omega - used to define the appropriate simple harmonic oscillator basis.
      Returns:
        The optimal dimension for modelling the system for appropriate accuracy of the model.'''
    
    # Using dimensionless units
    hbar = 1/(8.575*10 **(-34)/ (1.0545718 * 10**(-34)))
    dim = 20
    
    # build simple harmonic oscillator basis
    x_operator = np.sqrt(hbar / (2 * mass * omega)) * (raising_operator(dim) + lowering_operator(dim)) 
    p_operator = 1j * np.sqrt(hbar * mass * omega / 2) * (raising_operator(dim) - lowering_operator(dim))
    
    # build system Hamiltonian
    h_sys = 1/(2 * mass) * p_operator * p_operator + V2 * x_operator * x_operator * x_operator * x_operator - V1 * x_operator * x_operator
    
    # build initial state
    q_init = 0
    if T == 0:
        p_init = 0
    else: 
        p_init = norminv(0.95, 0, np.sqrt(T))
    
    # defining the initial wave function
    init_wave = init_coherent_state(dim, p_init, q_init, mass, omega)
    
    # computing the eigenvalues of the system hamiltonian
    e_levels, e_eigen = np.linalg.eig(h_sys)
    
    # prob of init state being in last used sho basis
    prob_of_last_state = (init_wave[-1, 0] * init_wave[-1, 0].conj()).real
    
    # define percentage change of 50th eigenvalue
    per_level_50 = 1
    
    # current energy level of 20th eigenstate
    eigval_of_int = sorted(e_levels.real, key=int)[19]
    
    # error in position after few iterations
    error_time_iter = 1
    np.random.seed(100)
    results = operator_time_average(init_wave, h_sys, h_sys * 0, x_operator, 0, time_step, 100, 1, "rk")
    pos_val = results[0][-1]
    
    #set up arrays for plots
    dim_array = [20]
    error_time_iter_array = [error_time_iter]
    eigval_of_int_array = [eigval_of_int]
    prob_array  = [prob_of_last_state]    
    while per_level_50 > 0.000001 or prob_of_last_state > 10 ** -50 or dim < 60 or error_time_iter > 10 ** -10:
        dim += 1
        # build simple harmonic oscillator basis
        x_operator = np.sqrt(hbar / (2 * mass * omega)) * (raising_operator(dim) + lowering_operator(dim)) 
        p_operator = 1j * np.sqrt(hbar * mass * omega / 2) * (raising_operator(dim) - lowering_operator(dim))
    
        # build system Hamiltonian
        h_sys = 1/(2 * mass) * p_operator * p_operator + V2 * x_operator * x_operator * x_operator * x_operator - V1 * x_operator * x_operator
    
        # build initial state
        q_init = 1
        if T == 0:
            p_init = 0
        else: 
            p_init = norminv(0.95, 0, np.sqrt(T))
    
        # defining the initial wave function
        init_wave = init_coherent_state(dim, p_init, q_init, mass, omega)
    
        # computing the eigenvalues of the system hamiltonian
        e_levels, e_eigen = np.linalg.eig(h_sys)
    
        # prob of init state being in last used sho basis
        prob_of_last_state = (init_wave[-1, 0] * init_wave[-1, 0].conj()).real
        
        # define percentage change of 50th eigenvalue
        per_level_50 = abs(eigval_of_int - sorted(e_levels.real, key=int)[19])/sorted(e_levels.real, key=int)[19]
        
        # current energy level of 50th eigenstate
        eigval_of_int = sorted(e_levels.real, key=int)[19]
        
        # computing the position value after 100 iterations
        results = operator_time_average(init_wave, h_sys, h_sys * 0, x_operator, 0, time_step, 200, 1, "rk")
        error_time_iter = abs(pos_val - results[0][-1])/pos_val
        pos_val = results[0][-1]
        dim_array.append(dim)
        error_time_iter_array.append(error_time_iter)
        eigval_of_int_array.append(eigval_of_int)
        prob_array.append(prob_of_last_state)
    return(dim, omega, dim_array, error_time_iter_array, eigval_of_int_array, prob_array)  