# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 18:52:17 2018

@author: Kamil

Script to analyse classical transition rates and transmission rates.
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import os
import datetime

print(datetime.datetime.now())
### For saving figures and clearing up any file paths
file_path = './Images/'
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

#### Varying parameters
T = 0.01
gamma = 0.1

#### Load all of the necessary operators
try:
    #### Individual trajectory of SSE equation
    data_path = './Data/classical_transition_rate_simulation_' + \
                   '_t' + str(T).replace(".", "") + '_g' + str(gamma).replace(".", "") + '.mat'
    matdata = scipy.io.loadmat(data_path)
    time_array = np.matrix(matdata['time_array'])
    trans_t = np.matrix(matdata['trans_t'])
    transmission_t = np.matrix(matdata['transmission_t'])
    correlation_t = np.matrix(matdata['correlation_t'])
except Exception as e:
    raise('Data has to be generated first')

plt.figure()
plt.plot(time_array.tolist()[0][1:], trans_t.tolist()[0][1:])
plt.ylabel(r'$k_{50000}^c(t)$')
plt.xlabel('t')
x1,x2,y1,y2 = plt.axis()
plt.axis((x1,x2,-0.008,0.008))
plt.tight_layout()
plt.show()
plt.savefig(file_path + 'transition_rate_c_t' + str(T).replace(".", "") +  '_g' + str(gamma).replace(".", "") + '.pdf')

plt.figure()
plt.plot(time_array.tolist()[0][1:], transmission_t.tolist()[0][1:])
plt.ylabel(r'$\kappa_{50000}^c(t)$')
plt.xlabel('t')
x1,x2,y1,y2 = plt.axis()
plt.axis((x1,x2,-1.1,1.1))
plt.tight_layout()
plt.show()
plt.savefig(file_path + 'transmission_rate_c_t' + str(T).replace(".", "") +  '_g' + str(gamma).replace(".", "") + '.pdf')