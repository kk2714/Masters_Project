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

from animator import *
from quantum_functions_library import *
import numpy as np
import matplotlib.pyplot as plt
import numba
import timeit
from matplotlib import animation
import numpy as np
import scipy.io

### Parameters of model - for my project these will remain constant, but can 
### be adjusted if required to look at other problems.

### v1 is the x^2 coefficient in V(x)
### v2 is the x^4 coefficient in V(x)
V1 = 1.5
V2 = 0.75
mass = 1
hbar = 1/(8.575*10 **(-34)/ (1.0545718 * 10**(-34)))
kb = 1
omega = 1
dim = 80

#project_A = projection_A(dim, mass, omega, hbar)
#
#### Get rid of nan values for project_A. This is due to numbers becoming too small
#### and having to replace the nan values with zero
#nan_values = np.isnan(project_A)
#project_A[nan_values] = 0
#
#print("checkpoint1")
#
#### Creating the x-grid projection operators
xmin = -3
xmax = 3
#no_of_points = 300
#x_grid_projectors, x_grid_midpoints = grid_projectors(xmin, xmax, no_of_points, dim, mass, omega, hbar)

infrastructure_file = 'qm_tools' + '_dim_' + str(dim).replace(".", "") + '_mass_' + str(mass).replace(".", "") +\
                      '_omega_' + str(omega).replace(".", "") + '.mat'
scipy.io.savemat(infrastructure_file, 
                 mdict={'x_grid_projectors': x_grid_projectors,
                        'x_grid_midpoints': x_grid_midpoints,
                        'x_limits': [xmin, xmax],
                        'project_A': project_A}, 
                 oned_as='row')
matdata = scipy.io.loadmat(infrastructure_file)
assert np.all(x_grid_projectors == matdata['x_grid_projectors'])
assert np.all(x_grid_midpoints == matdata['x_grid_midpoints'])
assert np.all(project_A == matdata['project_A'])
print("checkpoint2")