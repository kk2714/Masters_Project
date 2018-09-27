# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 17:41:59 2018

@author: Kamil

Comments: Script for visualising individual realisations of classical path trajectories.
"""

import classical_functions_library as cfl
import matplotlib.pyplot as plt

### For saving figures
file_path = './Images/'

### Note could automate this more by defining parameters at the top, but every
### figure had to be adjusted manually so did not make much sense! Potential FIX
### ME for further automation.

T = 0.25
gamma = 0.25
t_max = 500
time_step = 0.001 
q_init = 1.0 
v_init = 0.1
## Figure 2.6, page 29
np.random.seed(52)
fig_26 = cfl.model_langevin(t_max, time_step, q_init, v_init, T, gamma)
plt.figure()
plt.plot(fig_26[0][::100], fig_26[1][::100])
plt.ylabel('q')
plt.xlabel('t')
x1,x2,y1,y2 = plt.axis()
plt.axis((x1,x2,-1.6,1.6))
plt.show()
plt.savefig(file_path + 'path_c_t' + str(T).replace(".", "") +  '_g' + str(gamma).replace(".", "") + '.pdf')

Y = fig_26[2][::100]
X = fig_26[1][::100]
plt.figure()
plt.quiver(X[:-1], Y[:-1], X[1:]-X[:-1], Y[1:]-Y[:-1])
plt.ylabel('p')
plt.xlabel('q')
x1,x2,y1,y2 = plt.axis()
plt.axis((-1.6,1.6,-1.6,1.6))
plt.show()
plt.savefig(file_path + 'phase_space_c_t' + str(T).replace(".", "") +  '_g' + str(gamma).replace(".", "") + '.pdf')