# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 23:46:30 2018

@author: Kamil

Library of all of the quantum methods used. Define them here and call them in 
inidividual scripts for clean scripting. 
"""

import numpy as np
import matplotlib.pyplot as plt
import numba
from numba import prange
from math import factorial
from scipy.stats import norm
import time as time_clock
from numpy.polynomial.hermite import *
from scipy.integrate import quad
import math
from scipy import linalg

# Simulating quantum system using stochastic Schrodinger equation, numba compiler
# and different numerical methods

# Define inverse of Gaussian distribution
def norminv(percentile, mean, stddev):
    return norm.ppf(percentile, loc = mean, scale = stddev)

# Define exponential of a matrix
def exp_matrix(matrix):
    '''Function computes the exponential of a matrix.
    Arguments:
        matrix - as np.matrix
    Outputs:
        exp_matrix - as np.matrix'''
    matrix = np.array(matrix)
    exp_matrix = linalg.expm(matrix)
    exp_matrix = np.matrix(exp_matrix)
    return exp_matrix

# Define lowering operator
def lowering_operator(dim):
    '''Function computes the lowering operator for SHO basis. 
      Arguments:
        dim - as integer. Dimension of NxN matrix
      Returns:
        lowering operator (NxN matrix) for simple harmonic oscillator basis'''
    A = np.matrix([[0 for x in range(dim)] for y in range(dim)], dtype = complex)
    for i in range(dim - 1):
        A[i, i + 1] = np.sqrt(i + 1)
    A[dim - 1, dim - 1] = 1.0
    return A
        
# Define raising operator
def raising_operator(dim):
    '''Function computes the raising operator for SHO basis.
      Arguments:
        dim - as integer. Dimension of NxN matrix
      Returns:
        raising operator (NxN matrix) for simple harmonic oscillator basis'''
    A = np.matrix([[0 for x in range(dim)] for y in range(dim)], dtype = complex)
    for i in range(1, dim):
        A[i, i - 1] = np.sqrt(i)
    return A

# Defining initial coherent state
def init_coherent_state(dim, p_init, q_init, mass, omega, hbar):
    '''Function computes the initial coherent state from the given arguments. For
       further information about coherent states refer to write up.
      Arguments:
        dim - as integer. Dimension of N matrix
        p_init - as float. Initial momentum of particle
        q_init - as float. Initial position of particle
        mass - as float. Mass of particle
        omega - as float natural frequency of the simple harmonic oscillator basis.
        hbar - as float.
      Returns:
        A coherent state (N matrix) for simple harmonic oscillator basis'''
    z = np.sqrt(mass * omega / (2 * hbar)) * q_init + 1j * 1/np.sqrt(2 * mass * omega * hbar) * p_init
    wavefunction = np.zeros((1, dim), dtype = complex)
    for j in range(dim):
        wavefunction[:, j] = np.exp(-0.5 * z * z.conj()) * z ** j / np.sqrt(float(factorial(j)))
    wavefunction = np.matrix(wavefunction, dtype = complex).T
    
    # Renormalise to account for lack of higher order states
    wavefunction = wavefunction/np.linalg.norm(wavefunction)
    return wavefunction
    
# Define a function to compute the expectation of an operator
@numba.jit
def expectation(operator, wavefunction):
    '''Function takes an operator and wavefunction and returns the expectation value.
      Arguments:
        operator - as np.matrix
        wavefunction - as np.matrix
      Returns:
        Expectation as a float'''
    wave_dagger = wavefunction.getH()
    exp_num_matrix = wave_dagger * operator * wavefunction
    exp_den_matrix = wave_dagger * wavefunction
    expectation = exp_num_matrix[0,0]/exp_den_matrix[0,0]
    return expectation

def herm_fun_mult(x, n, m, mass, omega, hbar):
    ''' Function returns a function phi_n(x) * phi_m(x), where phi_k is the k'th
    simple harmonic oscillator (SHO) basis function.
    Arguments:
        x - as float, position
        n, m - as integers, n'th, m'th SHO basis functions of interest
        mass - as float, mass of particle
        omega - as float, natural frequency of SHO
        hbar - as float, rescaled hbar
    Output:
        phi_n(x)*phi_m(x) - as float.'''
    herm_coeff_n = np.zeros(n+1)
    herm_coeff_n[n] = 1
    herm_coeff_m = np.zeros(m+1)
    herm_coeff_m[m] = 1
    phi_n = np.exp(-mass * omega * x ** 2/(2 * hbar)) * hermval((mass * omega / hbar) ** 0.5 * x, herm_coeff_n)
    phi_n = np.multiply(phi_n, 1 / (math.pow(2, n) * math.factorial(n)) ** 0.5 * (mass * omega / (np.pi * hbar)) ** 0.25)
    phi_m = np.exp(-mass * omega * x ** 2/(2 * hbar)) * hermval((mass * omega / hbar) ** 0.5 * x, herm_coeff_m)
    phi_m = np.multiply(phi_m, 1 / (math.pow(2, m) * math.factorial(m)) ** 0.5 * (mass * omega / (np.pi * hbar)) ** 0.25)
    return phi_n * phi_m
    
# Integrating \int_{-\infinity}^{0} \phi_n * \phi m dx
def integrate_herm_fun_mult_A(n, m, mass, omega, hbar):
    ''' Function returns the integral of function phi_n(x) * phi_m(x) in the interval [-inf, 0],
    where phi_k is the k'th simple harmonic oscillator (SHO) basis function.
    Arguments:
        n, m - as integers, n'th, m'th SHO basis functions of interest
        mass - as float, mass of particle
        omega - as float, natural frequency of SHO
        hbar - as float, rescaled hbar
    Output:
        integral - as float.'''
    return quad(herm_fun_mult, -np.inf, 0, args=(n, m, mass, omega, hbar))[0]    

# Integrating within specified interval [xmin, xmax]: \int_{xmin}^{xmax} \phi_n * \phi m dx
def integrate_herm_fun_mult_interval(xmin, xmax, n, m, mass, omega, hbar):
    ''' Function returns the integral of function phi_n(x) * phi_m(x) in the interval [-inf, 0],
    where phi_k is the k'th simple harmonic oscillator (SHO) basis function.
    Arguments:
        xmin - as float, minimum x for integral
        xmax - as float, maximum x for integral
        n, m - as integers, n'th, m'th SHO basis functions of interest
        mass - as float, mass of particle
        omega - as float, natural frequency of SHO
        hbar - as float, rescaled hbar
    Output:
        integral - as float.'''
    return quad(herm_fun_mult, xmin, xmax, args=(n, m, mass, omega, hbar))[0]    

# Integrating \int_{0}^{-infinity} \phi_n * \phi m dx
def integrate_herm_fun_mult_B(n, m, mass, omega, hbar):
    ''' Function returns the integral of function phi_n(x) * phi_m(x) in the interval [0, inf],
    where phi_k is the k'th simple harmonic oscillator (SHO) basis function.
    Arguments:
        n, m - as integers, n'th, m'th SHO basis functions of interest
        mass - as float, mass of particle
        omega - as float, natural frequency of SHO
        hbar - as float, rescaled hbar
    Output:
        projection_operator - as np.matrix'''
    return quad(herm_fun_mult, 0, np.inf, args=(n, m, mass, omega, hbar))[0]    

# Defining projection operator P_A, projecting onto being in state A, x in (-inf, 0)
def projection_A(dim, mass, omega, hbar):
    ''' Function returns the projection of a wavefunction of being in the interval [-inf, 0].
    Arguments:
        dim - as integer, dimension of Hilbert space
        mass - as float, mass of particle
        omega - as float, natural frequency of SHO
        hbar - as float, rescaled hbar
    Output:
        projection_operator - as np.matrix.'''
    operator = np.matrix([[0 for x in range(dim)] for y in range(dim)], dtype = complex)
    for n in range(0, dim):
        for m in range(0, dim):
            operator[n, m] = integrate_herm_fun_mult_A(n, m, mass, omega, hbar)
    return operator

def projection_position(dim, x, mass, omega, hbar):
    ''' Function returns the projection onto position x, i.e \ket{x}\bra{x} operator.
        For further information refer to write up.
    Arguments:
        dim - as integer, dimension of the Hilbert space
        mass - as float, mass of particle,
        omega - as float, natural frequency of SHO
        hbar - as float, rescaled hbar
        x - as float, only real values
    Output:
        projection_position - as np.matrix'''
    operator = np.matrix([[0 for x in range(dim)] for y in range(dim)], dtype = complex)
    for n in range(0, dim):
        for m in range(0, dim):
            operator[n, m] = herm_fun_mult(x, n, m, mass, omega, hbar)
    return operator

# Computing trace of a matrix
def trace_matrix(matrix):
    ''' Function returns the trace of the matrix in SHO basis.
    Arguments:
        matrix - as np.matrix. Matrix over, which trace should be performed.
    Output:
        trace as float.'''
    n = matrix.shape[0]
    trace = 0
    for j in range(n):
        sho_base = np.matrix([0 for x in range(n)], dtype = complex)
        sho_base[0, j] = 1
        trace += sho_base * matrix * sho_base.T
    return trace[0, 0].real
    
# Defining projection operator P_B, projecting onto being in state B, x in (0, inf)
def projection_B(dim, mass, omega, hbar):
    ''' Function returns the projection of a wavefunction of being in the interval [0, inf].
    Arguments:
        dim - as integer, dimension of Hilbert space
        mass - as float, mass of particle
        omega - as float, natural frequency of SHO
        hbar - as float, rescaled hbar'''
    operator = np.matrix([[0 for x in range(dim)] for y in range(dim)], dtype = complex)
    for n in range(0, dim):
        for m in range(0, dim):
            operator[n, m] = integrate_herm_fun_mult_B(n, m, mass, omega, hbar)
    return operator    

# Defining grid projectors for an input, xmin, xmax, no. of points. Helpful in later plotting prob dist.
def grid_projectors(xmin, xmax, no_of_points, dim, mass, omega, hbar):
    ''' Function returns the projection operators for the coordinates [x_i, x_{i+1}]. Interval width
    delta x = x_{i+1} - x_1 determined by no_of_points specified.
    Arguments:
        xmin - as float, the smallest x-value for grid
        xmax - as float, the largest x-value for grid
        no_of_points - as integer, the number of points in the given interval
        dim - as integer, dimension of Hilbert space
        mass - as float, mass of particle
        omega - as float, natural frequency of SHO
        hbar - as float, rescaled hbar
    Output:
        operators - Python list with each element being a projection operator for interval [x_i, x_{i+1}].
        mid_points - Python list of midpoints of each interval [x_i, x_{i+1}]'''
    x_grid = np.linspace(xmin, xmax, num = no_of_points)
    operators = []
    mid_points = []
    for k in range(0, len(x_grid) - 1):
        operator = np.matrix([[0 for x in range(dim)] for y in range(dim)], dtype = complex)
        for n in range(0, dim):
            for m in range(0, dim):
                operator[n, m] = integrate_herm_fun_mult_interval(x_grid[k], x_grid[k+1], n, m, mass, omega, hbar)
        operators.append(operator)
        mid_points.append((x_grid[k] + x_grid[k+1])/2)
    return operators, mid_points

@numba.jit
def drift_coeff(wavefunction, hamiltonian, lindblad, hbar):
    ''' Function returns the drift term needed for evaluation of stochastic
    Schrodinger equation in numerical schemes.
      Arguments:
        wavefunction - N matrix defining the current state. 
        hamiltonian - NxN matrix defining the Hamiltonian of the system for the Lindblad equation
        lindblad - NxN matrix defining the Lindblad operator for Lindblad equation.
        hbar - as float. 
      Returns:
        The drift term needed for stochastic Schrodinger equation in numerical schemes.'''
    drift = -1j/hbar * hamiltonian * wavefunction + expectation(lindblad.getH(), wavefunction) * lindblad * wavefunction - \
            1/2 * lindblad.getH() * lindblad * wavefunction - 1/2 * expectation(lindblad.getH(), wavefunction) * expectation(lindblad, wavefunction) * wavefunction
    return drift

@numba.jit
def diffusion_term(wavefunction, hamiltonian, lindblad, hbar):
    '''Function returns the diffusion term needed for evaluation of stochastic
    Schrodinger equation in numerical schemes.
       Arguments:
        wavefunction - N matrix defining the current state. 
        hamiltonian - NxN matrix defining the Hamiltonian of the system for the Lindblad equation
        lindblad - NxN matrix defining the Lindblad operator for Lindblad equation.
        hbar - as float.
      Returns:
        The diffusion term needed for stochastic Schrodinger equation in numerical schemes.'''
    diffusion = lindblad * wavefunction - expectation(lindblad, wavefunction) * wavefunction
    return diffusion

# Implementing Euler scheme. Section 7.2.2 of "Theory of Open Quantum Systems". 
# Bruer and Petruccione

@numba.jit
def simulate_sse_euler(init_state, hamiltonian, lindblad, hbar, time_step, no_time_steps):
    '''Arguments:
        init_state - N matrix defining initial state. Initial state will be propagated in time
        hamiltonian - NxN matrix defining the Hamiltonian of the system for the Lindblad equation
        lindblad - NxN matrix defining the Lindblad operator for Lindblad equation.
        hbar - as float. 
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
        drift = drift_coeff(wave_prev, hamiltonian, lindblad, hbar)
        diffusion = diffusion_term(wave_prev, hamiltonian, lindblad, hbar)
        wave_after =  wave_prev + drift * time_step + diffusion * np.sqrt(time_step) * np.random.normal(0,1)
        wave_after = np.asarray(wave_after).reshape(-1)
        wave_evol[:, k] = wave_after/np.linalg.norm(wave_after)
        k += 1
    return wave_evol

# Implementing Heun scheme. Section 7.2.3 of "Theory of Open Quantum Systems". 
# Bruer and Petruccione
    
@numba.jit
def simulate_sse_heun(init_state, hamiltonian, lindblad, hbar, time_step, no_time_steps):
    '''Arguments:
        init_state - N matrix defining initial state. Initial state will be propagated in time
        hamiltonian - NxN matrix defining the Hamiltonian of the system for the Lindblad equation
        lindblad - NxN matrix defining the Lindblad operator for Lindblad equation.
        hbar - as float. 
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
        diffusion = diffusion_term(wave_step_before, hamiltonian, lindblad, hbar)
        
        # Computing Wiener increment
        wiener = np.sqrt(time_step) * np.random.normal(0,1)
        
        # Computing additional wavefunction and drifts
        drift_1 = drift_coeff(wave_step_before, hamiltonian, lindblad, hbar)
        wave_step_interm = wave_step_before + drift_1 * time_step + diffusion * wiener
        drift_2 = drift_coeff(wave_step_interm, hamiltonian, lindblad, hbar)
        
        # Final wavefunction
        wave_step_after =  wave_step_before + 0.5 * (drift_1 + drift_2) * time_step + diffusion * wiener
        wave_step_after = np.asarray(wave_step_after).reshape(-1)
        wave_evol[:, k] = wave_step_after/np.linalg.norm(wave_step_after)
        k += 1
    return wave_evol

# Implementing Runge-Kutta scheme. Section 7.2.4 of "Theory of Open Quantum Systems". 
# Bruer and Petruccione
    
@numba.jit
def simulate_sse_rk(init_state, hamiltonian, lindblad, hbar, time_step, no_time_steps):
    '''Arguments:
        init_state - N matrix defining initial state. Initial state will be propagated in time
        hamiltonian - NxN matrix defining the Hamiltonian of the system for the Lindblad equation
        lindblad - NxN matrix defining the Lindblad operator for Lindblad equation.
        hbar - as float. 
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
        diffusion = diffusion_term(wave_step_before, hamiltonian, lindblad, hbar)
        
        # Computing intermediate wavefunction and drifts
        drift_1 = drift_coeff(wave_step_before, hamiltonian, lindblad, hbar)
        wave_interm_1 = wave_step_before + 0.5 * time_step * drift_1
        
        drift_2 = drift_coeff(wave_interm_1, hamiltonian, lindblad, hbar)
        wave_interm2 = wave_step_before + 0.5 * time_step * drift_2
        
        drift_3 = drift_coeff(wave_interm2, hamiltonian, lindblad, hbar)
        wave_interm3 = wave_step_before + time_step * drift_3
        
        drift_4 = drift_coeff(wave_interm3, hamiltonian, lindblad, hbar)
        
        drift_overall = 1/6 * (drift_1 + 2 * drift_2 + 2 * drift_3 + drift_4)
        
        # Final wavefunction
        wave_step_after =  wave_step_before + drift_overall * time_step + diffusion * np.sqrt(time_step) * np.random.normal(0,1)
        wave_step_after = np.asarray(wave_step_after).reshape(-1)
        wave_evol[:, k] = wave_step_after/np.linalg.norm(wave_step_after)
        k += 1
    return wave_evol

# Implementing Platen scheme. Section 7.2.5 of "Theory of Open Quantum Systems". 
# Bruer and Petruccione
    
@numba.jit
def simulate_sse_platen(init_state, hamiltonian, lindblad, hbar, time_step, no_time_steps):
    '''Arguments:
        init_state - N matrix defining initial state. Initial state will be propagated in time
        hamiltonian - NxN matrix defining the Hamiltonian of the system for the Lindblad equation
        lindblad - NxN matrix defining the Lindblad operator for Lindblad equation.
        hbar - as float. 
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
        diffusion = diffusion_term(wave_step_before, hamiltonian, lindblad, hbar)
        drift = drift_coeff(wave_step_before, hamiltonian, lindblad, hbar)
        
        # Computing Wiener increment
        wiener = np.sqrt(time_step) * np.random.normal(0,1)
        
        # Computing intermediate wavefunction
        wave_interm_hat = wave_step_before + drift * time_step + diffusion * wiener
        wave_interm_pos = wave_step_before + drift * time_step + diffusion * np.sqrt(time_step)
        wave_interm_neg = wave_step_before + drift * time_step - diffusion * np.sqrt(time_step)
        
        # Computing the remaining necessary drift terms
        drift_hat = drift_coeff(wave_interm_hat, hamiltonian, lindblad, hbar)
        
        diffusion_pos = diffusion_term(wave_interm_pos, hamiltonian, lindblad, hbar)
        diffusion_neg = diffusion_term(wave_interm_neg, hamiltonian, lindblad, hbar)
        
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
    return wave_evol

# Define a function to compute an operator time-average with standard error
@numba.jit(parallel=True)
def operator_time_average(init_state, hamiltonian, lindblad, operator, hbar, time_step, no_time_steps, no_of_realisations, method):
    '''Function computes the time evolution of an operator by averaging over a given number of
    realisations of individual solutions of the stochastic Schrodinger equation with a specified
    initial wavefunction via a chosen numerical scheme. Currently support solving the SSE via Euler, 
    Runge-Kutta, Heun and Platen methods.
       Arguments:
        init_state - N matrix defining initial state. Initial state will be propagated in time
        hamiltonian - NxN matrix defining the Hamiltonian of the system for the Lindblad equation
        lindblad - NxN matrix defining the Lindblad operator for Lindblad equation.
        operator - NxN matrix. Operator, whos time average we are interested in.
        hbar - as float. 
        time_step - as float. The time step used in the numerical scheme.
        no_time_steps - as integer. The amount of time steps to propagate through.
        no_of_realisations - as float. No of realisations to average over.
        method - as string. Valid inputs include: "euler", "rk", "heun", "platen". 
      Returns:
        The average over realisation for the time-dependent operator. Additionally returns the standard
        error on the result as a matrix'''
    results = np.zeros((no_time_steps, no_of_realisations), dtype = float)
    time = np.arange(0, no_time_steps * time_step, time_step, dtype = float)
    wave_evol_matrix = np.zeros((init_state.shape[0], len(time), no_of_realisations), dtype = complex)
    if method not in ("euler", "rk", "heun", "platen"):
        raise ValueError("Unknown numerical scheme. Please read function specification.")    
    # Euler method
    if method == "euler":
        for j in range(no_of_realisations):
            wave_evol = simulate_sse_euler(init_state, hamiltonian, lindblad, hbar, time_step, no_time_steps)
            wave_evol_matrix[:, :, j] = wave_evol
            for k in range(wave_evol.shape[1]):
                wave_func = np.matrix(wave_evol[:, k], dtype = complex).T
                results[k, j] += expectation(operator, wave_func)
    # Heun method
    if method == "heun":
        for j in range(no_of_realisations):
            wave_evol = simulate_sse_heun(init_state, hamiltonian, lindblad, hbar, time_step, no_time_steps)
            wave_evol_matrix[:, :, j] = wave_evol
            for k in range(wave_evol.shape[1]):
                wave_func = np.matrix(wave_evol[:, k], dtype = complex).T
                results[k, j] += expectation(operator, wave_func)
    # Runge-Kutta method
    if method == "rk":
        for j in range(no_of_realisations):
            wave_evol = simulate_sse_rk(init_state, hamiltonian, lindblad, hbar, time_step, no_time_steps)
            wave_evol_matrix[:, :, j] = wave_evol
            for k in range(wave_evol.shape[1]):
                wave_func = np.matrix(wave_evol[:, k], dtype = complex).T
                results[k, j] += expectation(operator, wave_func)
    # Platen method
    if method == "platen":
        for j in range(no_of_realisations):
            wave_evol = simulate_sse_platen(init_state, hamiltonian, lindblad, hbar, time_step, no_time_steps)
            wave_evol_matrix[:, :, j] = wave_evol
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
    
    ### To be able to keep the wavefunction as well just in case
    return time, average, standard_error, wave_evol_matrix

# Define a function to compute two time-averages of operator with standard error.
# Designed with the intention of using it for position and momentum operators, but
# other operators equally valid. Additionally, keep all data as creating it is CPU
# extensive.
    
### Parallelised processing does not work. FIX ME! Might have to fix the whole script.
@numba.jit(parallel=True)
def two_operator_time_average_all_data(init_state, hamiltonian, lindblad, operator1, operator2, hbar, time_step, no_time_steps, no_of_realisations, method):
    '''Function computes the time evolution of two operators by averaging over a given number of
    realisations of individual solutions of the stochastic Schrodinger equation with a specified
    initial wavefunction via a chosen numerical scheme. Currently support solving the SSE via Euler, 
    Runge-Kutta, Heun and Platen methods.
       Arguments:
        init_state - N matrix defining initial state. Initial state will be propagated in time
        hamiltonian - NxN matrix defining the Hamiltonian of the system for the Lindblad equation
        lindblad - NxN matrix defining the Lindblad operator for Lindblad equation.
        operator1 - NxN matrix. Operator, whos time average we are interested in.
        operator2 - NxN matrix. Operator, whos time average we are interested in.
        hbar - as float. 
        time_step - as float. The time step used in the numerical scheme.
        no_time_steps - as integer. The amount of time steps to propagate through.
        no_of_realisations - as float. No of realisations to average over.
        method - as string. Valid inputs include: "euler", "rk", "heun", "platen". 
      Returns:
        The average over realisation for the time-dependent operator. Additionally returns the standard
        error on the result as a matrix'''
    results1 = np.zeros((no_time_steps, no_of_realisations), dtype = float)
    results2 = np.zeros((no_time_steps, no_of_realisations), dtype = float)
    time = np.arange(0, no_time_steps * time_step, time_step, dtype = float)
    wave_evol_matrix = np.zeros((init_state.shape[0], len(time), no_of_realisations), dtype = complex)
    if method not in ("euler", "rk", "heun", "platen"):
        raise ValueError("Unknown numerical scheme. Please read function specification.")    
    # Euler method
    if method == "euler":
        for j in range(no_of_realisations):
            wave_evol = simulate_sse_euler(init_state, hamiltonian, lindblad, hbar, time_step, no_time_steps)
            wave_evol_matrix[:, :, j] = wave_evol
            for k in range(wave_evol.shape[1]):
                wave_func = np.matrix(wave_evol[:, k], dtype = complex).T
                results1[k, j] += expectation(operator1, wave_func)
                results2[k, j] += expectation(operator2, wave_func)
    # Heun method
    if method == "heun":
        for j in range(no_of_realisations):
            wave_evol = simulate_sse_heun(init_state, hamiltonian, lindblad, hbar, time_step, no_time_steps)
            wave_evol_matrix[:, :, j] = wave_evol
            for k in range(wave_evol.shape[1]):
                wave_func = np.matrix(wave_evol[:, k], dtype = complex).T
                results1[k, j] += expectation(operator1, wave_func)
                results2[k, j] += expectation(operator2, wave_func)
    # Runge-Kutta method
    if method == "rk":
        for j in range(no_of_realisations):
            if j%100 == 0:
                print(j)
            wave_evol = simulate_sse_rk(init_state, hamiltonian, lindblad, hbar, time_step, no_time_steps)
            wave_evol_matrix[:, :, j] = wave_evol
            for k in range(wave_evol.shape[1]):
                wave_func = np.matrix(wave_evol[:, k], dtype = complex).T
                results1[k, j] += expectation(operator1, wave_func)
                results2[k, j] += expectation(operator2, wave_func)
    # Platen method
    if method == "platen":
        for j in range(no_of_realisations):
            wave_evol = simulate_sse_platen(init_state, hamiltonian, lindblad, hbar, time_step, no_time_steps)
            wave_evol_matrix[:, :, j] = wave_evol
            for k in range(wave_evol.shape[1]):
                wave_func = np.matrix(wave_evol[:, k], dtype = complex).T
                results1[k, j] += expectation(operator1, wave_func)
                results2[k, j] += expectation(operator2, wave_func)
    average1 = np.mean(results1, axis = 1)
    average2 = np.mean(results2, axis = 1)
    # Computing the standard error on expectation value
    if no_of_realisations != 1 and no_of_realisations > 0:
        standard_error1 = np.zeros((no_time_steps, no_of_realisations), dtype = float)
        standard_error2 = np.zeros((no_time_steps, no_of_realisations), dtype = float)
        for j in range(results1.shape[1]):
            standard_error1[:, j] = results1[:, j] - average1
            standard_error2[:, j] = results2[:, j] - average2
        standard_error1 = np.multiply(standard_error1, standard_error1)
        standard_error1 = np.sum(standard_error1, axis = 1)
        standard_error1 = standard_error1/(len(standard_error1) * (len(standard_error1) - 1))
        standard_error1 = np.sqrt(standard_error1)
        standard_error2 = np.multiply(standard_error2, standard_error2)
        standard_error2 = np.sum(standard_error2, axis = 1)
        standard_error2 = standard_error2/(len(standard_error2) * (len(standard_error2) - 1))
        standard_error2 = np.sqrt(standard_error2)
    else:
        standard_error1 = np.zeros(no_time_steps, dtype = float)
        standard_error2 = np.zeros(no_time_steps, dtype = float)
    
    ### To be able to keep the wavefunction as well just in case
    return time, average1, standard_error1, average2, standard_error2, results1, results2, wave_evol_matrix

# Defining Lindblad drift based on Liouville equation
@numba.jit
def lindblad_drift(rho, hamiltonian, lindblad, hbar):
    '''Arguments:
        rho - NxN matrix defining the current rho state. 
        hamiltonian - NxN matrix defining the Hamiltonian of the system for the Lindblad equation
        lindblad - NxN matrix defining the Lindblad operator for Lindblad equation.
        hbar - as float. 
      Returns:
        The drift term needed for Liouville equation.'''
    drift = - 1j/hbar * (hamiltonian * rho - rho * hamiltonian) + \
           lindblad * rho * lindblad.getH() - 0.5 * rho * lindblad.getH() * lindblad - \
           0.5 * lindblad.getH() * lindblad * rho
    return drift

### Integrating Lindblad directly
@numba.jit
def int_lindblad_liouv_euler(init_state, hamiltonian, lindblad, hbar, time_step, no_steps):
    '''Arguments:
        init_state - N matrix defining initial state.
        hamiltonian - NxN matrix defining the Hamiltonian of the system for the Lindblad equation
        lindblad - NxN matrix defining the Lindblad operator for Lindblad equation.
        hbar - as float
        time_step - as float. The time step used in the numerical scheme.
        no_steps - as integer. The amount of time steps to propagate through.
      Returns:
        The density operator at different times propagated according to the Lindblad equation.'''
    n = init_state.shape[0]
    rho_evol = np.empty((n, n, no_steps), dtype = complex)
    rho_init = init_state * init_state.getH()
    rho_evol[:, :, 0] = np.asarray(rho_init)
    for i in range(0, no_steps - 1):
        rho_evol[:, :, i+1] = rho_evol[:, :, i] + lindblad_drift(rho_evol[:, :, i], hamiltonian, lindblad, hbar) * time_step
        rho_evol[:, :, i+1] = rho_evol[:, :, i+1] / trace_matrix(rho_evol[:, :, i+1])
    return rho_evol

### Obtaining the expectation value of operator in time by using Euler scheme to integrate Lindblad equation.
### Unstable in time. Below have a method for explicit integration for time independent Lindblad operators!
@numba.jit
def tr_rho_operator_euler(init_rho, operator, hamiltonian, lindblad, hbar, time_step, no_steps):
    '''Arguments:
        init_rho - NxN matrix defining initial state.
        operator - NxN matrix, whose expectation value we are interested in
        hamiltonian - NxN matrix defining the Hamiltonian of the system for the Lindblad equation
        lindblad - NxN matrix defining the Lindblad operator for Lindblad equation.
        hbar - as float
        time_step - as float. The time step used in the numerical scheme.
        no_steps - as integer. The amount of time steps to propagate through.
      Returns:
        The expectation value of operator changing in time as an array'''
    tr_rho_operator = np.empty(no_steps, dtype = complex)
    rho_evol0 = init_rho
    rho_evol1 = rho_evol0 + lindblad_drift(rho_evol0, hamiltonian, lindblad, hbar) * time_step
    rho_evol1 = rho_evol1 / trace_matrix(rho_evol1)
    tr_rho_operator[0] = trace_matrix(operator * rho_evol0)
    tr_rho_operator[1] = trace_matrix(operator * rho_evol1)
    for i in range(2, no_steps):
        rho_evol0 = rho_evol1
        rho_evol1 = rho_evol0 + lindblad_drift(rho_evol0, hamiltonian, lindblad, hbar) * time_step
        rho_evol1 = rho_evol1 / trace_matrix(rho_evol1)
        tr_rho_operator[i] = trace_matrix(operator * rho_evol1)
        if abs(trace_matrix(rho_evol1)) > 2 or abs(trace_matrix(operator * rho_evol1)) > 2:
            break
    return tr_rho_operator


#### Improvement on above Euler method. Can use superoperator method to directly
#### solve the Lindblad equation. The details of the mathematical method are also provided
### in the pdf file.
@numba.jit
def rho_lindblad_superoperator(init_rho, hamiltonian, lindblad, hbar, time_step, no_steps):
    '''Arguments:
        init_rho - NxN matrix defining initial state.
        hamiltonian - NxN matrix defining the Hamiltonian of the system for the Lindblad equation
        lindblad - NxN matrix defining the Lindblad operator for Lindblad equation.
        hbar - as float
        time_step - as float. The time step used in the numerical scheme.
        no_steps - as integer. The amount of time steps to propagate through.
      Returns:
        The expectation value of operator changing in time as an array.'''
    dim = init_rho.shape[0]
    identity = np.diag(np.ones(dim, dtype=complex))
    rho_operator = np.empty((dim, dim, no_steps), dtype = complex)
    rho_operator[:, :, 0] = init_rho
    hamiltonian_liouv = -1j/hbar * (np.kron(hamiltonian, identity) - np.kron(identity, hamiltonian.T))
    lindblad_liouv = np.kron(lindblad, identity) * np.kron(identity, lindblad.getH().T) - 0.5 * np.kron(identity, (lindblad.getH() * lindblad).T) - 0.5 * np.kron(lindblad.getH() * lindblad, identity)
    liouv_operator = hamiltonian_liouv + lindblad_liouv
    rho_evol_liouv = np.reshape(init_rho, dim * dim, order = 'C').T
    propagator = exp_matrix(liouv_operator * time_step)
    for i in range(1, no_steps):
        rho_evol_liouv = propagator * rho_evol_liouv
        rho_evol_ham = np.reshape(rho_evol_liouv, (dim, dim), order = 'C')
        rho_evol_ham = rho_evol_ham / trace_matrix(rho_evol_ham)
        rho_operator[:, :, i] = rho_evol_ham
        rho_evol_liouv = np.reshape(rho_evol_ham, dim * dim, order = 'C').T
        if abs(trace_matrix(rho_evol_ham)) > 2:
            break
    return rho_operator

### Method defined to find the expectation value of an operator only. Does not keep
### the evolution of the density matrix.
@numba.jit
def exp_op_lindblad_superoperator(init_rho, operator, hamiltonian, lindblad, hbar, time_step, no_steps):
    '''Arguments:
        init_rho - NxN matrix defining initial state.
        operator - NxN matrix defining the operator, whose expecation value in time we are interested in
        hamiltonian - NxN matrix defining the Hamiltonian of the system for the Lindblad equation
        lindblad - NxN matrix defining the Lindblad operator for Lindblad equation.
        hbar - as float
        time_step - as float. The time step used in the numerical scheme.
        no_steps - as integer. The amount of time steps to propagate through.
      Returns:
        The expectation value of operator changing in time as an array.'''
    dim = init_rho.shape[0]
    identity = np.diag(np.ones(dim, dtype=complex))
    tr_rho_operator = np.empty(no_steps, dtype = complex)
    tr_rho_operator[0] = trace_matrix(operator * init_rho)
    hamiltonian_liouv = -1j/hbar * (np.kron(hamiltonian, identity) - np.kron(identity, hamiltonian.T))
    lindblad_liouv = np.kron(lindblad, identity) * np.kron(identity, lindblad.getH().T) - 0.5 * np.kron(identity, (lindblad.getH() * lindblad).T) - 0.5 * np.kron(lindblad.getH() * lindblad, identity)
    liouv_operator = hamiltonian_liouv + lindblad_liouv
    rho_evol_liouv = np.reshape(init_rho, dim * dim, order = 'C').T
    propagator = exp_matrix(liouv_operator * time_step)
    for i in range(1, no_steps):
        rho_evol_liouv = propagator * rho_evol_liouv
        rho_evol_ham = np.reshape(rho_evol_liouv, (dim, dim), order = 'C')
        rho_evol_ham = rho_evol_ham / trace_matrix(rho_evol_ham)
        tr_rho_operator[i] = trace_matrix(operator * rho_evol_ham)
        rho_evol_liouv = np.reshape(rho_evol_ham, dim * dim, order = 'C').T
        if abs(trace_matrix(rho_evol_ham)) > 2 or abs(trace_matrix(operator * rho_evol_ham)) > 2:
            break
    return tr_rho_operator

### Creates coherent states needed for evaluation of Husimi plots. Creates a grid,
### where the coherent states are evaluated.
@numba.jit
def husimi_coherent_states(dim, mass, omega, hbar, qmin, qmax, dq, pmin, pmax, dp):
    '''Function that takes the parameters of the toy model and creates the coherent states
    along with their data required for Husimi representation.
    Arguments:
        dim - as integer. The dimension of the Hilbert space.
        mass - as float. The mass of the particle.
        omega - as float. The natural frequency of SHO basis.
        hbar - as float. Rescaled dimensions.
        qmin - as float. Is the minimum value of q-coordinate
        qmax - as float. Is the maximum value of p-coordinate
        dq - as float. Is the resolution of q-axis
        pmin - as float. Is the minimum value of p-coordinate.
        pmax - as float. Is the maximum value of p-coordinate.
        dp - as float. Is the resolution of p-axis.
    Output:
        coherent_states - a list of matrices of dim N representing the coherent states
        coherent_states_data - a matrix containing all of the data about coherent states. 
                               The number of rows has to match the length of coherent states.
                               First column is the q_0 of coherent state, second is p_0, third is var(q),
                               fourth is var(p).'''
    #### Need to define some operators
    x_operator = np.sqrt(hbar / (2 * mass * omega)) * (raising_operator(dim) + lowering_operator(dim)) 
    p_operator = 1j * np.sqrt(hbar * mass * omega / 2) * (raising_operator(dim) - lowering_operator(dim)) 
    ###### Define mesh grid on phase space
    p_grid = np.arange(pmin, pmax + dp, dp)
    q_grid = np.arange(qmin, qmax + dq, dq)
    coherent_states = []
    coherent_states_data = np.zeros((len(p_grid) * len(q_grid), 4), dtype = float)
    i = 0
    for j in range(len(p_grid)):
        for k in range(len(q_grid)):
            potential_cs = init_coherent_state(dim, p_grid[j], q_grid[k], mass, omega, hbar)
            coherent_states.append(potential_cs)
            coherent_states_data[i, 0] = q_grid[k]
            coherent_states_data[i, 1] = p_grid[j]
            coherent_states_data[i, 2] = (potential_cs.getH() * p_operator * p_operator * potential_cs - (potential_cs.getH() * p_operator * potential_cs) ** 2)[0,0].real
            coherent_states_data[i, 3] = (potential_cs.getH() * x_operator * x_operator * potential_cs - (potential_cs.getH() * x_operator * potential_cs) ** 2)[0,0].real
            i += 1
    return coherent_states, coherent_states_data

@numba.jit
def gaussian_state_prob_distr_phase_space(q, p, q_average, p_average, q_var, p_var):
    prob = 1/(p_var * q_var) * np.exp(-(q - q_average) ** 2 / (2 * q_var)) * np.exp(-(p - p_average) ** 2 / (2 * p_var))
    return prob

### Function used to evaluate the Husimi plot for an individual wave function given
### the data on coherent states.    
### Probably can parallelize with Concurrent Futures. FIX ME.
@numba.jit
def quantum_phase_space_prob_distr(wave_func, coherent_states, coherent_states_data, qmin, qmax, dq, pmin, pmax, dp):
    '''Function that takes the initial wavefunction along with coherent states and parameters
    needed to define a phase space grid and produces a Husimi representation.
    Arguments:
        wave_func - an N matrix representing the wave function
        coherent_states - a list of matrices of dim N representing the coherent states
        coherent_states_data - a matrix containing all of the data about coherent states. 
                               The number of rows has to match the length of coherent states.
                               First column is the q_0 of coherent state, second is p_0, third is var(q),
                               fourth is var(p).
        qmin - as float. Is the minimum value of q-coordinate
        qmax - as float. Is the maximum value of p-coordinate
        dq - as float. Is the resolution of q-axis
        pmin - as float. Is the minimum value of p-coordinate.
        pmax - as float. Is the maximum value of p-coordinate.
        dp - as float. Is the resolution of p-axis.
    Output:
        prob_dist - as np.array of dimension (len(q_grid), len(p_grid)) with values of prob
                    distribution at those coordinates in phase space.
        q_grid - as np.array the q-grid needed for contour plot
        p_grid - as np.array the p-grid needed for contour plot
        coherent_state_weight - as np.array the weighting of each coherent state'''
    #### Define phase space grid
    p_grid = np.arange(pmin, pmax + dp, dp)
    q_grid = np.arange(qmin, qmax + dq, dq)
    prob_phase_space = np.zeros((len(q_grid), len(p_grid)), dtype = float)    
    
    #### Compute weighting of each coherent state
    weights_coh_states = np.zeros(len(coherent_states), dtype = float)
    for j in range(len(coherent_states)):
        weight = (coherent_states[j].getH() * wave_func)
        weight = (weight * weight.conjugate())[0, 0].real
        weights_coh_states[j] = weight
        #### Do mini grid straight away to avoid another for loop
        mini_grid = np.zeros((len(q_grid), len(p_grid)), dtype = float)
        for q_idx in range(len(q_grid)):
            for p_idx in prange(len(p_grid)):
                mini_grid[q_idx, p_idx] = gaussian_state_prob_distr_phase_space(q_grid[q_idx], 
                                                                                p_grid[p_idx], 
                                                                                coherent_states_data[j, 0], # the e[q]
                                                                                coherent_states_data[j, 1], # the e[p]
                                                                                coherent_states_data[j, 2], # the var[q]
                                                                                coherent_states_data[j, 3]) # the var[p]
        mini_grid = mini_grid / np.sum(mini_grid)
        prob_phase_space += weights_coh_states[j] * mini_grid
    weights_coh_states = weights_coh_states / np.sum(weights_coh_states)    
    prob_phase_space = prob_phase_space / np.sum(weights_coh_states)
    return prob_phase_space, q_grid, p_grid, weights_coh_states

#### Convergence function requires more work and bullet proofing. Unclear as to
#### which eigenstate convergence to choose. FIX ME.
#@numba.jit
#def convergence_function(V1, V2, hbar, T, mass, omega, time_step):
#    ''' The Hamiltonian is of the form: 0.5 p^2/m - v1 q^2 + v2 q^4. The initial state is a coherent
#    state dependepent on the temperature, T. This function computes the optimal dimension for modelling the 
#    open quantum dynamics of the system based on the simple harmonic oscillator basis. 
#      
#    Arguments:
#        v1, v2 - floats. Used to describe the quartic energy potential. 
#        gamma - as float. "Friction" coefficient in the open quantum system.
#        T - temperature of the system
#        mass - mass of the particle.
#        omega - used to define the appropriate simple harmonic oscillator basis.
#      Returns:
#        The optimal dimension for modelling the system for appropriate accuracy of the model.'''
#    
#    # defining initial dimension of Hilbert space
#    dim = 20
#    
#    # build simple harmonic oscillator basis
#    x_operator = np.sqrt(hbar / (2 * mass * omega)) * (raising_operator(dim) + lowering_operator(dim)) 
#    p_operator = 1j * np.sqrt(hbar * mass * omega / 2) * (raising_operator(dim) - lowering_operator(dim))
#    
#    # build system Hamiltonian
#    h_sys = 1/(2 * mass) * p_operator * p_operator + V2 * x_operator * x_operator * x_operator * x_operator - V1 * x_operator * x_operator
#    
#    # build initial state
#    q_init = 0
#    if T == 0:
#        p_init = 0
#    else: 
#        p_init = norminv(0.95, 0, np.sqrt(T))
#    
#    # defining the initial wave function
#    init_wave = init_coherent_state(dim, p_init, q_init, mass, omega, hbar)
#    
#    # computing the eigenvalues of the system hamiltonian
#    e_levels, e_eigen = np.linalg.eig(h_sys)
#    
#    # prob of init state being in last used sho basis
#    prob_of_last_state = (init_wave[-1, 0] * init_wave[-1, 0].conj()).real
#    
#    # define percentage change of 50th eigenvalue
#    per_level_50 = 1
#    
#    # current energy level of 20th eigenstate
#    eigval_of_int = sorted(e_levels.real, key=int)[19]
#    
#    # error in position after few iterations
#    error_time_iter = 1
#    np.random.seed(100)
#    results = operator_time_average(init_wave, h_sys, h_sys * 0, x_operator, 0, time_step, 100, 1, "rk")
#    pos_val = results[0][-1]
#    
#    #set up arrays for plots
#    dim_array = [20]
#    error_time_iter_array = [error_time_iter]
#    eigval_of_int_array = [eigval_of_int]
#    prob_array  = [prob_of_last_state]    
#    while per_level_50 > 0.000001 or prob_of_last_state > 10 ** -50 or dim < 60 or error_time_iter > 10 ** -10:
#        dim += 1
#        # build simple harmonic oscillator basis
#        x_operator = np.sqrt(hbar / (2 * mass * omega)) * (raising_operator(dim) + lowering_operator(dim)) 
#        p_operator = 1j * np.sqrt(hbar * mass * omega / 2) * (raising_operator(dim) - lowering_operator(dim))
#    
#        # build system Hamiltonian
#        h_sys = 1/(2 * mass) * p_operator * p_operator + V2 * x_operator * x_operator * x_operator * x_operator - V1 * x_operator * x_operator
#    
#        # build initial state
#        q_init = 1
#        if T == 0:
#            p_init = 0
#        else: 
#            p_init = norminv(0.95, 0, np.sqrt(T))
#    
#        # defining the initial wave function
#        init_wave = init_coherent_state(dim, p_init, q_init, mass, omega, hbar)
#    
#        # computing the eigenvalues of the system hamiltonian
#        e_levels, e_eigen = np.linalg.eig(h_sys)
#    
#        # prob of init state being in last used sho basis
#        prob_of_last_state = (init_wave[-1, 0] * init_wave[-1, 0].conj()).real
#        
#        # define percentage change of 50th eigenvalue
#        per_level_50 = abs(eigval_of_int - sorted(e_levels.real, key=int)[19])/sorted(e_levels.real, key=int)[19]
#        
#        # current energy level of 50th eigenstate
#        eigval_of_int = sorted(e_levels.real, key=int)[19]
#        
#        # computing the position value after 100 iterations
#        results = operator_time_average(init_wave, h_sys, h_sys * 0, x_operator, 0, time_step, 200, 1, "rk")
#        error_time_iter = abs(pos_val - results[0][-1])/pos_val
#        pos_val = results[0][-1]
#        dim_array.append(dim)
#        error_time_iter_array.append(error_time_iter)
#        eigval_of_int_array.append(eigval_of_int)
#        prob_array.append(prob_of_last_state)
#    return dim, omega, dim_array, error_time_iter_array, eigval_of_int_array, prob_array