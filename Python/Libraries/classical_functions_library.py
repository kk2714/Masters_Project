# -*- coding: utf-8 -*-
"""
Created on Thur Apr  19 19:45:30 2018

@author: Kamil

Comments:
    
Library of all of the classical methods used. Define them here and call them in 
inidividual scripts for clean scripting. 

Creating a function to model Langevin equation using the Verlet
algorithm with dissipative and noise terms. Based on: Second-order integrators 
for Langevin equations with holonomic constraints by Eric Vanden-Eijnden and
Giovanni Ciccotti
"""

import numpy as np
import matplotlib.pyplot as plt
import numba
from scipy.stats import norm
from scipy import integrate

def norminv(percentile, mean, stddev):
    '''Function for obtaining the inverse of a Gaussian distribution.
    Arguments:
        percentile - as a float
        mean - as a float
        stddev - as a float
    Returns:
        The inverse of Gaussian distribution as float.'''
    return norm.ppf(percentile, loc = mean, scale = stddev)


@numba.jit
def hamiltonian_evaluation(q, p, V1, V2, mass):
    ''' Function that evaluates the classical Hamiltonian expression H=p^2/2m - V1 q^2 + V2 q^4.
    Arguments:
        q - as a float. The position coordinate
        p - as a float. The momentum
        mass - as a float. Mass of particle.
        V1 - as a float. The coefficient in V(q) = - V1 q^2 + V2 q^4
        V2 - as a float. The coefficient in V(q) = - V1 q^2 + V2 q^4
    Returns:
        H(q,p) - as a float evaluated for given parameters'''
    hamiltonian = p ** 2 /(2 * mass) - V1 * q ** 2 + V2 * q ** 4
    return hamiltonian

@numba.jit
def exponential_hamiltonian_evaluation(q, p, V1, V2, mass):
    ''' Function that evaluates the exponential of Hamiltonian expression 
    exp(-H) = exp (-(p^2/2m - V1 q^2 + V2 q^4)).
    Arguments:
        q - as a float. The position coordinate
        p - as a float. The momentum
        mass - as a float. Mass of particle.
        V1 - as a float. The coefficient in V(q) = - V1 q^2 + V2 q^4
        V2 - as a float. The coefficient in V(q) = - V1 q^2 + V2 q^4
    Returns:
        exp(-H(q,p)) - as a float evaluated for given parameters'''
    exp_ham = np.exp(-hamiltonian_evaluation(q, p, V1, V2, mass))
    return exp_ham

@numba.jit
def classical_partition_function(V1, V2, mass):
    '''Function that evaluates the classical partition function defined as
    Z = \int_{-infty}^infty \int_{-infty}^infty exp(-H(q,p)). With the Hamiltonian
    defined as
    H=p^2/2m - V1 q^2 + V2 q^4.
    Arguments:
        mass - as a float. Mass of particle.
        V1 - as a float. The coefficient in V(q) = - V1 q^2 + V2 q^4
        V2 - as a float. The coefficient in V(q) = - V1 q^2 + V2 q^4
    Returns:
        Z - as a float evaluated for given parameters'''
    Z = integrate.nquad(exponential_hamiltonian_evaluation, [[-np.inf, np.inf], [-np.inf, np.inf]], args=(V1, V2, mass,))[0]
    return Z

@numba.jit
def acceleration(position, mass):
    '''Function for obtaining the acceleration due to the quartic potential of 
    the form V(x) = V1 x^2 + V2 x^4. As the potential under consideration will
    not change V1 and V2 have been hard coded. Acceleration is -dV/dx
    Arguments:
        position - as a float
        mass - as a float
    Returns:
        The acceleration depending on the position in the potential for particle
        of mass, m. Returns a float.'''
    V1 = 1.5
    V2 = 0.75
    
    # Computing the value of acceleration
    accel = (2 * V1 * position - 4 * V2 * position ** 3) / mass
    return accel
    
@numba.jit
def model_langevin(t_max, time_step, q_init, v_init, T, gamma):
    '''Function for modelling the path of a classical particle following Langevin
    equation. The discretisation is based on the Verlet algorithm (second order
    accurate). 
    ***Note using rescaled dimensions with kb = 1, mass = 1. Refer to 
    write up for further guidance.
    Arguments:
        t_max - as a float. The time up to which integrating.
        time_step - as a float. The smaller the better the accuracy.
        q_init - as a float. The initial position.
        v_init - as a float. The initial velocity.
        T - as a float. The temperature.
        gamma - as a float. The coupling coefficient.
    Returns:
        time_points(t) - as array. The time array.
        position(t) - as array. The position of the particle at discrete times tk.
        momentum(t) - as array. The momentum of the particle at discrete times tk.'''
    # Specifying all of the physical parameters
    kb = 1
    mass = 1
    lamda = np.sqrt(2 * mass * T * kb * gamma)
    sigma = lamda / mass
    
    # Creating all of the necessary matrices
    time_points = np.arange(0, t_max, time_step, dtype = float)
    q_t = np.zeros(len(time_points), dtype = float)
    v_t = np.zeros(len(time_points), dtype = float)
    
    # initialising position and momentum
    q_t[0] = q_init
    v_t[0] = v_init

    # implementing the method
    for i in range(len(time_points)):
        phi = np.random.normal(0,1)
        theta = np.random.normal(0,1)
        A = 0.5 * time_step ** 2 * (acceleration(q_t[i], mass) - gamma * v_t[i]) + \
            sigma/2 * time_step ** 1.5 * (phi + theta/np.sqrt(3))
        q_t[i+1] = q_t[i] + time_step * v_t[i] + A
        v_t[i+1] = v_t[i] + 0.5 * time_step * (acceleration(q_t[i], mass) + acceleration(q_t[i+1], mass)) - \
                   time_step * gamma * v_t[i] + np.sqrt(time_step) * sigma * phi - gamma * A
    
    p_t = v_t * mass
    return time_points, q_t, p_t

# Langevin model for transition/transmission rate to manage RAM limits more efficiently
@numba.jit
def model_langevin_tr(t_max, time_step, q_init, v_init, T, gamma):
    '''Function for modelling the path of a classical particle following Langevin
    equation for the computation of the transition/transmission rate. In 
    transition/transmission rate averaging
    out over 100,000s of individual trajectories and have to manage RAM limits.
    Function returns truncated position and momentum arrays. Chosen to return 
    every 20th element of original arrays.
    
    ***Note using rescaled dimensions with kb = 1, mass = 1. Refer to 
    write up for further guidance.
    Arguments:
        t_max - as a float. The time up to which integrating.
        time_step - as a float. The smaller the better the accuracy.
        q_init - as a float. The initial position.
        v_init - as a float. The initial velocity.
        T - as a float. The temperature.
        gamma - as a float. The coupling coefficient.
    Returns:
        time_points(t) - as array. The time array.
        position(t) - as array. The position of the particle at discrete times tk.
        momentum(t) - as array. The momentum of the particle at discrete times tk.'''
    # Specifying all of the physical parameters
    kb = 1
    mass = 1
    lamda = np.sqrt(2 * mass * T * kb * gamma)
    sigma = lamda / mass
    
    # Creating all of the necessary matrices
    time_points = np.arange(0, t_max, time_step, dtype = float)
    q_t = np.zeros(len(time_points), dtype = float)
    v_t = np.zeros(len(time_points), dtype = float)
    
    # initialising position and momentum
    q_t[0] = q_init
    v_t[0] = v_init

    # implementing the method
    for i in range(len(time_points)):
        phi = np.random.normal(0,1)
        theta = np.random.normal(0,1)
        A = 0.5 * time_step ** 2 * (acceleration(q_t[i], mass) - gamma * v_t[i]) + \
            sigma/2 * time_step ** 1.5 * (phi + theta/np.sqrt(3))
        q_t[i+1] = q_t[i] + time_step * v_t[i] + A
        v_t[i+1] = v_t[i] + 0.5 * time_step * (acceleration(q_t[i], mass) + acceleration(q_t[i+1], mass)) - \
                   time_step * gamma * v_t[i] + np.sqrt(time_step) * sigma * phi - gamma * A
    
    ret_time_points = time_points[0::20]
    ret_q_t = q_t[0::20]
    ret_p_t = v_t[0::20] * mass
    
    return ret_time_points, ret_q_t, ret_p_t

# Creating a function to compute the transmission rate using Eq. 2.66 found in 
# Chiara Liverani's PhD thesis. Creating a function to compute the transition 
# rate using Eq provided in thesis write up

@numba.jit(parallel=True)
def transition_rate(no_of_realis, T, gamma, time_step, t_max):
    '''Function for computing the transition rate 
    1/no_of_real sqrt(2pi kb T/mass) \sum_{0}^K p_init \theta(q_t)|q_0 = q*. 
    For further details on the equation refer to write up. 
    Transition rate computed by modelling K realisations of a classical particle
    moving in the energy potential governed by the Langevin equation. Averaging 
    carried out over the initial conditions. The initial velocity is randomly 
    chosen from the Maxwell-Boltzmann distribution and the particles initial 
    starting position is at q_0=0.
    
    ***Note using rescaled dimensions with kb = 1, mass = 1. Refer to 
    write up for further guidance.
    Arguments:
        no_of_realis - as integer. No of realisations to average over.
        T - as a float. The temperature.
        gamma - as a float. The coupling coefficient.
        time_step - as a float. The smaller the better the accuracy.
        t_max - as a float. The time up to which integrating.
    Returns:
        time_points(t) - as array. The time array.
        transition_rate(t) - as array. The transition rate at discrete times tk.'''
    # Specifying all physical parameters
    mass = 1
    kb = 1
    
    # Computing the partition function - hard code in the values for now.
    Z = classical_partition_function(1.5, 0.75, 1.0)
    
    # Doing the first instance to know the size of the array
    percentile = np.random.rand()
    p_init = norminv(percentile, 0, np.sqrt(T))
    q_init = 0
    model = model_langevin_tr(t_max, time_step, q_init, p_init, T, gamma)
    # Initialise arrays
    time_points = model[0]
    q_t = np.zeros((len(time_points), no_of_realis), dtype = float)
    p_t = np.zeros((len(time_points), no_of_realis), dtype = float)
    trans_t = np.zeros(len(time_points), dtype = float) 
    correlation_t = np.zeros(len(time_points), dtype = float)
    transmission_t = np.zeros(len(time_points), dtype = float)
    q_t[:, 0] = model[1] 
    p_t[:, 0] = model[2]
    
    for i in range(1, no_of_realis):
        percentile = np.random.rand()
        p_init = norminv(percentile, 0, np.sqrt(T))
        q_init = 0
        model = model_langevin_tr(t_max, time_step, q_init, p_init, T, gamma)
        q_t[:, i] = model[1] 
        p_t[:, i] = model[2]
    
    q_t_raw = q_t    
    for i in range(no_of_realis):
        for j in range(len(time_points)):
            if q_t[j, i] >= 0:
                q_t[j, i] = 1
            else:
                q_t[j, i] = 0
    
    for i in range(len(time_points)):
        ### evaluating the transition rate
        summation = p_t[0, :] * q_t[i, :]
        trans_t[i] = (1/no_of_realis) * (2/(mass * Z)) * np.sum(summation)
        transmission_t[i] = (1/no_of_realis) * np.sqrt(2 * np.pi/(kb * mass * T)) * np.sum(summation)
        
        ### evaluating the correlation function for exponential decay
        correlation_t[i] = (1/no_of_realis) * np.sum(q_t[i, :])
        
    return time_points, trans_t, transmission_t, correlation_t, q_t_raw, p_t