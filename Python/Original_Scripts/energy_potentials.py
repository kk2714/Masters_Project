# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 23:51:13 2018

@author: Kamil
"""
import numba
import numpy as np
import matplotlib.pyplot as plt
from math import factorial

### For saving figures
file_path = 'C://Users/Kamil/Dropbox/Masters Project/Write up/Images/'

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
def init_coherent_state(dim, p_init, q_init, mass, omega, hbar):
    '''Arguments:
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
    return(wavefunction)

@numba.jit
def energy_potential(q_points, v1, v2):
    q_squared = np.multiply(q_points, q_points)
    fx = v2 * np.multiply(q_squared, q_squared) - v1 * q_squared
    return fx

@numba.jit
def sho_energy_potential(q_points, mass, omega):
    fx = 0.5 * mass * omega ** 2 * np.multiply(q_points, q_points)
    return fx

@numba.jit
def eig_energy_ham(V1, V2, hbar, mass, omega):
    '''Shows the convergence of eigenstates and eigenenergies of Hamiltonian with
    increasing dimension of matrix.
    Inputs:
        V1 - as float. the q^2 coefficient.
        V2 - as float. the q^4 coefficient.
        hbar - as float.
        T - as float. Temperature.
        mass - as float. Mass of particle
        omega - defines SHO potential.
    Outputs:
        eigenstates - 3d array showing convergence of eigenstates with increasing dim of Hilbert space.
        eigenenergies - 3d array showing convergence of eigenenergies with increasing dim of Hilbert space.'''
        
    first_state = []
    second_state = []
    third_state = []
    fourth_state = []
    fifth_state = []
    tenth_state = []
    for j in range(20, 120):
        # build simple harmonic oscillator basis
        x_operator = np.sqrt(hbar / (2 * mass * omega)) * (raising_operator(j) + lowering_operator(j)) 
        p_operator = 1j * np.sqrt(hbar * mass * omega / 2) * (raising_operator(j) - lowering_operator(j))
    
        # build system Hamiltonian
        h_sys = 1/(2 * mass) * p_operator * p_operator + V2 * x_operator * x_operator * x_operator * x_operator - V1 * x_operator * x_operator
        e_levels, e_eigen = np.linalg.eig(h_sys)
        # due to shape of potential energies are degenerate!
        first_state.append(sorted(e_levels.real)[0])
        second_state.append(sorted(e_levels.real)[2])
        third_state.append(sorted(e_levels.real)[4])
        fourth_state.append(sorted(e_levels.real)[6])
        fifth_state.append(sorted(e_levels.real)[8])
        tenth_state.append(sorted(e_levels.real)[19])
    dim = range(20,120)
    return(dim, first_state, second_state, third_state, fourth_state, fifth_state, tenth_state)

q_points = np.arange(-1.5, 1.5, 0.001, dtype = float)
e_points = energy_potential(q_points, 1.5, 0.75)
sho_e_points = sho_energy_potential(q_points, 1, 0.5)
plt.figure()
plt.plot(q_points, e_points)
plt.plot(q_points, sho_e_points)
plt.legend((r'V(q)', r'$V_{SHO}(q)$'),
           shadow=True, loc=(0.70, 0.80))
plt.ylabel('E(q)')
plt.xlabel('q')
plt.show()

# Define parameters of the system
V1 = 1.5
V2 = 0.75
mass = 1
hbar = 1/(8.575*10 **(-34)/ (1.0545718 * 10**(-34)))
omega = 1 

## Compute prob of being in each state
#x_operator = np.sqrt(hbar / (2 * mass * omega)) * (raising_operator(120) + lowering_operator(120)) 
#p_operator = 1j * np.sqrt(hbar * mass * omega / 2) * (raising_operator(120) - lowering_operator(120))
#    
## build system Hamiltonian
#h_sys = 1/(2 * mass) * p_operator * p_operator + V2 * x_operator * x_operator * x_operator * x_operator - V1 * x_operator * x_operator
#init_wave = init_coherent_state(120, 0, 1, mass, omega, hbar)
#e_levels, e_eigen = np.linalg.eig(h_sys)
#idx = e_levels.argsort()[::-1]   
#e_levels = e_levels[idx]
#e_eigen = e_eigen[:,idx]
#prob_first = np.conjugate((e_eigen[0] * init_wave)[0,0]) * ((e_eigen[0] * init_wave)[0,0])
#prob_second = np.conjugate((e_eigen[2] * init_wave)[0,0]) * ((e_eigen[2] * init_wave)[0,0])
#prob_third = np.conjugate((e_eigen[4] * init_wave)[0,0]) * ((e_eigen[4] * init_wave)[0,0])
#prob_fourth = np.conjugate((e_eigen[6] * init_wave)[0,0]) * ((e_eigen[6] * init_wave)[0,0])
#
#prob_random = np.conjugate((e_eigen[20] * init_wave)[0,0]) * ((e_eigen[20] * init_wave)[0,0])
#
#print(prob_first)
#print(prob_second)
#print(prob_third)
#print(prob_fourth)
#print(prob_random)
#
## Sketch code
#prob_total = 0
#highest_prob = 0
#for j in range(e_eigen.shape[0]):
#    prob_state = np.conjugate((e_eigen[j] * init_wave)[0,0]) * ((e_eigen[j] * init_wave)[0,0])
#    prob_total += np.conjugate((e_eigen[j] * init_wave)[0,0]) * ((e_eigen[j] * init_wave)[0,0])
#    if prob_state > highest_prob:
#        highest_prob = prob_state
#        print(j)
#    
#print(prob_total)

### Plot the V(q) - hey
x = np.arange(-1.5, 1.5, 0.001)
v_x = -1.5 * x ** 2 + 0.75 * x ** 4
plt.figure()
plt.plot(x, v_x)
plt.ylabel(r'V(q)')
plt.xlabel('q')
plt.annotate('State A', xy = (-1.2, -0.25), xycoords='data', fontsize = 15)
plt.annotate('State B', xy = (0.70, -0.25), xycoords='data', fontsize = 15)
#plt.annotate(r'$E_B = 0.75$', xy = (-0.20, -0.5), xycoords='data', fontsize = 12)
#plt.annotate("", xy=(0.0, -0.01), xycoords='data', xytext=(0.0, -0.75),
#                            textcoords='data',
#                            va="center", ha="center",
#                            arrowprops=dict(arrowstyle="|-|",
#                                            connectionstyle="arc3,rad=0"))
plt.axvline(x=0, ymin=0, ls = 'dashed')
x1,x2,y1,y2 = plt.axis()
plt.axis((x1,x2,-0.8,0.5))
plt.show()
plt.savefig(file_path + 'double_well_potential' + '.pdf')
                                

## Plotting the change in the 10th energy
#eig_e = eig_energy_ham(V1, V2, hbar, mass, omega)
#plt.figure()
#plt.plot(eig_e[0], eig_e[6], 'r--', linewidth = 2.0)
#plt.ylabel(r'$E_{10}$')
#plt.xlabel('N')
#plt.figure()
#q_points = np.arange(-1.5, 1.5, 0.001, dtype = float)
#e_points = energy_potential(q_points, 1.5, 0.75)
#plt.plot(q_points, e_points)
#plt.axhline(y=eig_e[1][-1], linewidth=2, color='r')
#plt.axhline(y=eig_e[2][-1], linewidth=2, color='g')
#plt.axhline(y=eig_e[3][-1], linewidth=2, color='y')
#plt.axhline(y=eig_e[4][-1], linewidth=2, color='m')
#plt.ylabel('E(q)')
#plt.xlabel('q')
#plt.legend(('V(q)', r'$E_0$', r'$E_1$', r'$E_2$', r'$E_3$'),
#           shadow=True, loc=(0.80, 0.65))
#
#ang_f = (eig_e[2][-1] - eig_e[1][-1])/hbar
#print("angular frequency is")
#print(ang_f)