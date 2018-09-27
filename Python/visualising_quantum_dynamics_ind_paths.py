# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 18:52:17 2018

@author: Kamil

Script for visualising the individual realisations of stochastic Schrodinger 
equation saved in .mat file.
"""

import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import scipy.io
import sys, os
import datetime

print(datetime.datetime.now())
### For saving figures and clearing up any file paths
file_path = './Images/'
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

### Initial conditions
p_init = 0.1 
q_init = 1
dim = 80
hbar = 1/(8.575*10 **(-34)/ (1.0545718 * 10**(-34)))
mass = 1
omega = 1

### Phase space grid
qmin = -3
qmax = 3
pmin = -2
pmax = 2

#### Varying parameters
T = 0.01
gamma = 0.001 / 2
init_state_str = 'coherent_state_' + 'q' + str(q_init).replace(".", "") + 'p_init' + str(p_init).replace(".", "")

#### Load all of the necessary operators
try:
    #### Individual trajectory of SSE equation
    data_path = './Data/qm_sse_simulation_ind_path_' + init_state_str + \
                  '_t' + str(T).replace(".", "") + '_g' + str(gamma).replace(".", "") + '.mat'
    matdata = scipy.io.loadmat(data_path)
    time_array = np.matrix(matdata['time_array'])
    x_operator_time_average = np.matrix(matdata['x_operator_time_average'])
    x_time_standard_error = np.matrix(matdata['x_time_standard_error'])
    p_operator_time_average = np.matrix(matdata['p_operator_time_average'])
    p_time_standard_error = np.matrix(matdata['p_time_standard_error'])
    wave_evol_matrix = np.matrix(matdata['wave_evol_matrix'])
except Exception as e:
    raise('Data has to be generated first')

plt.figure()
plt.plot(time_array.tolist()[0], x_operator_time_average.tolist()[0])
plt.ylabel(r'$\langle q \rangle$')
plt.xlabel('t')
x1,x2,y1,y2 = plt.axis()
plt.axis((x1,x2,-1.6,1.6))
plt.annotate('Point A',
            xy=(74.429, 0.666),
            xycoords='data',
            xytext=(-15,
                    -50),
            arrowprops=dict(arrowstyle='->'),
            textcoords='offset points')
plt.annotate('Point B',
            xy=(175.336, 0.961),
            xycoords='data',
            xytext=(-15,
                    -50),
            arrowprops=dict(arrowstyle='->'),
            textcoords='offset points')
plt.show()
plt.savefig(file_path + 'path_q_t' + str(T).replace(".", "") +  '_g' + str(gamma).replace(".", "") + '.pdf')

Y = np.asarray(p_operator_time_average.tolist()[0][::100])
X = np.asarray(x_operator_time_average.tolist()[0][::100])
plt.figure()
plt.quiver(X[:-1], Y[:-1], X[1:]-X[:-1], Y[1:]-Y[:-1])
plt.ylabel(r'$\langle p \rangle$')
plt.xlabel(r'$\langle q \rangle$')
x1,x2,y1,y2 = plt.axis()
plt.axis((-1.6,1.6,-1.6,1.6))
plt.show()
plt.savefig(file_path + 'phase_space_expectation_q_t' + str(T).replace(".", "") +  '_g' + str(gamma).replace(".", "") + '.pdf')