# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 21:10:41 2018

@author: Kamil

Creating all of the necessary operators and storing them as a Matlab file to later
not have to generate the same data over and over again.
"""
import sys, os
this_file_path = os.getcwd()
sys.path.append(this_file_path.replace("\Data", ""))

import numpy as np
import scipy.io
import quantum_functions_library as qfl

#### Parameters of model that will not change    
mass = 1
hbar = 1/(8.575*10 **(-34)/ (1.0545718 * 10**(-34)))
kb = 1
omega = 1
dim = 80
qmin = -3
qmax = 3
dq = 0.3
pmin = -2
pmax = 2
dp = 0.3

coherent_states, coherent_states_data = qfl.husimi_coherent_states(dim, mass, omega, hbar, qmin, qmax, dq, pmin, pmax, dp)

infrastructure_file = 'qm_tools_husimi' + '_dim_' + str(dim).replace(".", "") + '_mass_' + str(mass).replace(".", "") +\
                      '_omega_' + str(omega).replace(".", "") + '.mat'
scipy.io.savemat(infrastructure_file, 
                 mdict={'coherent_states': coherent_states,
                        'coherent_states_data': coherent_states_data}, 
                 oned_as='row')
matdata = scipy.io.loadmat(infrastructure_file)
assert np.all(coherent_states == matdata['coherent_states'])
assert np.all(coherent_states_data == matdata['coherent_states_data'])
print("checkpoint2")