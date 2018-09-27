# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 18:52:17 2018

@author: Kamil

Script to analyse quantum transition rates.
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

### Initial conditions
p_init = 0.1 
q_init = 1
dim = 80
hbar = 1/(8.575*10 **(-34)/ (1.0545718 * 10**(-34)))
mass = 1
omega = 1

#### Varying parameters
gamma = 0.1 / 2
T = 0.01
init_state_str = 'eq_state_t' + str(T).replace(".", "") + '_'

short_time_step = 0.1
short_no_steps = 500

#### Load all of the necessary operators
try:
    #### Individual trajectory of SSE equation
    data_path = './Data/qm_lindblad_simulation_' + init_state_str + \
                   '_t' + str(T).replace(".", "") + '_g' + str(gamma).replace(".", "") + '.mat'
    matdata = scipy.io.loadmat(data_path)
    short_time = np.matrix(matdata['short_time'])
    prob_in_short_time = np.matrix(matdata['prob_in_short_time'])
    proj_A_short_time = np.matrix(matdata['proj_A_short_time'])
    short_tran_rate_current_density = np.matrix(matdata['short_tran_rate_current_density'])
    
    data_xgrid_path = './Data/qm_tools_dim_' + str(dim).replace(".", "") + '_mass_' + str(mass).replace(".", "") + '_omega_' + \
                str(omega).replace(".", "") + '.mat'
    matdata = scipy.io.loadmat(data_xgrid_path)
    x_grid_midpoints = matdata['x_grid_midpoints']
    xmin = matdata['x_limits'][0][0]
    xmax = matdata['x_limits'][0][1]
except Exception as e:
    raise('Data has to be generated first')

dx = x_grid_midpoints[0][1] - x_grid_midpoints[0][0]
z = np.array(prob_in_short_time)
y, x = np.mgrid[slice(0, short_no_steps * short_time_step, short_time_step), 
                slice(xmin, xmax, dx)]
z_min, z_max = 0, np.abs(prob_in_short_time).max()
plt.figure()
plt.pcolormesh(x, y, z, cmap='RdBu', vmin=z_min, vmax=0.05)
### set the limits of the plot to the limits of the data
plt.axis([x.min(), x.max(), y.min(), y.max()])
col = plt.colorbar()
col.set_label(r'$|\psi(q)|^2$', rotation = 270, labelpad = 20)
plt.xlabel(r'$q$')
plt.ylabel(r'$t$')
plt.tight_layout()
plt.show()
plt.savefig(file_path + 'qm_contour_dm_evol_t' + str(T).replace(".", "") + '_gamma' + \
            str(gamma).replace(".", "") + '_' + init_state_str + 'long_time' + '.pdf')

plt.figure()
plt.plot(short_time.tolist()[0], np.log(4 * np.abs((np.array(proj_A_short_time.tolist()[0]) - 0.5))))
plt.ylabel(r'$ln(\frac{\langle P_A(t) \rangle - 0.5}{0.25})$')
plt.xlabel('t')
plt.tight_layout()
plt.savefig(file_path + 'qm_log_A_t' + str(T).replace(".", "") + '_gamma' + \
            str(gamma).replace(".", "") + '_' + init_state_str + 'long_time' + '.pdf')

print((np.log(4 * np.abs((np.array(proj_A_short_time.tolist()[0]) - 0.5)))[-1] - np.log(4 * np.abs((np.array(proj_A_short_time.tolist()[0]) - 0.5)))[100])/(short_time.tolist()[0][100]-short_time.tolist()[0][-1]))

plt.figure()
plt.plot(short_time.tolist()[0], short_tran_rate_current_density.tolist()[0])
plt.ylabel(r'$k^q(t)$')
plt.xlabel('t')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.tight_layout()
plt.savefig(file_path + 'qm_curr_den_tran_rate_t' + str(T).replace(".", "") + '_gamma' + \
            str(gamma).replace(".", "") + '_' + init_state_str + 'short_time' + '.pdf')

print(short_tran_rate_current_density.tolist()[0][-1])
