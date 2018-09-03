# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 17:52:05 2018

@author: Kamil
"""
import numpy as np

init_state_str = 'hey'
results = np.arange(10)
T = 0.1
gamma = 0.2

##### Save all of the data
data_output_path = './Data/qm_sse_simulation_ind_path_' + init_state_str + \
                   '_t' + str(T).replace(".", "") + '_g' + str(gamma).replace(".", "") + '.npy'

np.save(data_output_path, results)