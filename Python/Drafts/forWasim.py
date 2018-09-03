# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 23:05:22 2018

@author: Kamil
"""

import numpy as np
import matplotlib.pyplot as plt
import numba

@numba.jit
def mf_approx(t_max, time_step, sx_init, sy_init, sz_init, epsilon, v):
    
    # Creating all of the necessary matrices
    time_points = np.arange(0, t_max, time_step, dtype = float)
    sx_t = np.zeros(len(time_points), dtype = float)
    sy_t = np.zeros(len(time_points), dtype = float)
    sz_t = np.zeros(len(time_points), dtype = float)
    
    # initialising the three coordinates
    sx_t[0] = sx_init
    sy_t[0] = sy_init
    sz_t[0] = sz_init

    # implementing the method
    for i in range(len(time_points)):
        sx_t[i+1] = sx_t[i] - time_step * epsilon * sy_t[i]
        sy_t[i+1] = sy_t[i] + time_step * epsilon * sx_t[i] + time_step * v/4 * (1 - 4 * sz_t[i] - 12 * sz_t[i] ** 2)
        sz_t[i+1] = sz_t[i] + time_step * v * sy_t[i] 
    
    return([time_points, sx_t, sy_t, sz_t])

## Modelling fixed point. remains fixed :)
#model_fx_point = mf_approx(10, 0.01, 0, 0, -0.5, 1.0, 1.0)
#plt.figure()
#plt.plot(model_fx_point[0], model_fx_point[1])
#plt.ylabel(r'$s_x$')
#plt.xlabel('t')
#plt.show()
#
#plt.figure()
#plt.plot(model_fx_point[0], model_fx_point[2])
#plt.ylabel(r'$s_y$')
#plt.xlabel('t')
#plt.show()
#
#plt.figure()
#plt.plot(model_fx_point[0], model_fx_point[3])
#plt.ylabel(r'$s_z$')
#plt.xlabel('t')
#plt.show()
    
# Modelling the constraint. remains fixed :)
model_fx_point = mf_approx(10, 0.0001, 0.5, 0.0, 0.0, 1.0, 1.0)
plt.figure()
plt.plot(model_fx_point[0], model_fx_point[1])
plt.ylabel(r'$s_x$')
plt.xlabel('t')
plt.show()

plt.figure()
plt.plot(model_fx_point[0], model_fx_point[2])
plt.ylabel(r'$s_y$')
plt.xlabel('t')
plt.show()

plt.figure()
plt.plot(model_fx_point[0], model_fx_point[3])
plt.ylabel(r'$s_z$')
plt.xlabel('t')
plt.show()

sx = model_fx_point[1]
sy = model_fx_point[2]
sz = model_fx_point[3]

# Checking 18 is verified
lhs_18 = np.multiply(sx, sx) + np.multiply(sy, sy)
mat_1 = 0.25 * (1 - 2 * sz)
mat_2 = 1 + 2 * sz
rhs_18 = np.multiply(np.multiply(mat_1, mat_2), mat_2)
plt.figure()
plt.plot(lhs_18, rhs_18)
plt.ylabel('lhs_18')
plt.xlabel('rhs_18')
plt.show()
