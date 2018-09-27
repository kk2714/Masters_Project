# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 14:01:16 2018

@author: Kamil
"""

import classical_functions_library as cfl
import matplotlib.pyplot as plt
import numpy as np
import datetime
import scipy.io

#### Varying parameters
T = 0.25
gamma = 0.25
p_init = 0.1 
q_init = 1
init_state_str = 'q' + str(q_init).replace(".", "") + 'p_init' + str(p_init).replace(".", "")
t_max = 500
time_step = 0.001
class_results = cfl.model_langevin(t_max, time_step, q_init, p_init, T, gamma)

##### Save all of the data
data_output_path = './Data/cd_langevin_simulation_ind_path_' + init_state_str + \
                   '_t' + str(T).replace(".", "") + '_g' + str(gamma).replace(".", "") + '.mat'

scipy.io.savemat(data_output_path, 
                 mdict={'time_array': class_results[0],
                        'q_values': class_results[1],
                        'p_values': class_results[2],
                        }, 
                 oned_as='row')
matdata = scipy.io.loadmat(data_output_path)
assert np.all(class_results[0] == matdata['time_array'])
assert np.all(class_results[1] == matdata['q_values'])
assert np.all(class_results[2] == matdata['p_values'])
print("checkpoint2")
print(datetime.datetime.now())

plt.figure()
plt.plot(class_results[0], class_results[1])
plt.figure()
plt.plot(class_results[1], class_results[2])