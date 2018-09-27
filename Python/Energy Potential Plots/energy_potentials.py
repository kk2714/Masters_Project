# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 23:51:13 2018

@author: Kamil

Script for visualising the energy potential.
"""
import numba
import numpy as np
import matplotlib.pyplot as plt

### For saving figures
file_path = './Images/'

@numba.jit
def energy_potential(q_points, v1, v2):
    q_squared = np.multiply(q_points, q_points)
    fx = v2 * np.multiply(q_squared, q_squared) - v1 * q_squared
    return fx

@numba.jit
def sho_energy_potential(q_points, mass, omega):
    fx = 0.5 * mass * omega ** 2 * np.multiply(q_points, q_points)
    return fx

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

### Plot the V(q)
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