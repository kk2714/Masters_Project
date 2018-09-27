# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 12:42:25 2018

@author: Kamil

Script to plot the initial distribution both as contour plot and 3D plot
using the Husimi functions.
"""
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import quantum_functions_library as qfl

file_path = './Images/'

#### Parameters of model that will not change    
### v1 is the x^2 coefficient
### v2 is the x^4 coefficient
V1 = 1.5
V2 = 0.75
mass = 1
hbar = 1/(8.575*10 **(-34)/ (1.0545718 * 10**(-34)))
kb = 1
omega = 1
dim = 80

#### Defining operators that will remain fixed
x_operator = np.sqrt(hbar / (2 * mass * omega)) * (qfl.raising_operator(dim) + qfl.lowering_operator(dim)) 
p_operator = 1j * np.sqrt(hbar * mass * omega / 2) * (qfl.raising_operator(dim) - qfl.lowering_operator(dim)) 
h_sys = 1/(2 * mass) * p_operator * p_operator + V2 * x_operator * x_operator * x_operator * x_operator - V1 * x_operator * x_operator

#### Varying parameters
T = 0.01
gamma = 0.001
# Initial conditions
percentile = np.random.rand()
p_init = 0.1 
q_init = 1
### Finding eigenvalues and eigenvectors of Hamiltonian of the system.
eigenenergies, eigenstates = np.linalg.eig(h_sys)
idx = eigenenergies.argsort()[::1]   
eigenenergies = eigenenergies[idx]
eigenstates = eigenstates[:,idx]

wave_func = qfl.init_coherent_state(dim, p_init, q_init, mass, omega, hbar)
name_of_state = 'coherent_state_'
#wave_func = eigenstates[0].T

qmin = -3
qmax = 3
dq = 0.3
pmin = -2
pmax = 2
dp = 0.3
coherent_states, coherent_states_data = qfl.husimi_coherent_states(dim, mass, omega, hbar, qmin, qmax, dq, pmin, pmax, dp)

dq = 0.025
dp = 0.025
prob_phase_space, q_grid, p_grid, weights_coh_states = qfl.quantum_phase_space_prob_distr(wave_func, coherent_states, coherent_states_data, qmin, qmax, dq, pmin, pmax, dp)

## Plot 1

z = prob_phase_space
q, p = np.mgrid[slice(qmin, qmax + dq, dq), 
                slice(pmin, pmax + dp, dp)]
z_min, z_max = 0, np.abs(prob_phase_space).max()
fig, ax = plt.subplots()
plt.pcolormesh(q, p, z, cmap='RdBu', vmin=z_min, vmax=z_max)
### set the limits of the plot to the limits of the data
plt.axis([q.min(), q.max(), p.min(), p.max()])
ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
col = plt.colorbar()
col.set_label(r'$|\psi(q,p)|^2$', rotation = 270, labelpad = 20)
plt.xlabel(r'$q$')
plt.ylabel(r'$p$')
fig.tight_layout()
plt.savefig(file_path + 'husimi_initial_dis_' + name_of_state + '_q' + str(q_init).replace(".", "") +  '_p' + str(p_init).replace(".", "") + '.pdf')

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(p, q, prob_phase_space, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

ax.set_xlabel('q')
ax.set_ylabel('p')
ax.yaxis.set_major_locator(plt.MaxNLocator(5))
ax.xaxis.set_major_locator(plt.MaxNLocator(5))
# Add a color bar which maps values to colors.
ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
col = fig.colorbar(surf, shrink=0.5, aspect=5)
col.set_label(r'$|\psi(q,p)|^2$', rotation = 270, labelpad = 20)
fig.tight_layout()
plt.savefig(file_path + 'husimi_initial_dis_3d_' + name_of_state + '_q' + str(q_init).replace(".", "") +  '_p' + str(p_init).replace(".", "") + '.pdf')
plt.show()