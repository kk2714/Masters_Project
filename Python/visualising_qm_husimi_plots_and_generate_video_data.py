# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 23:43:51 2018

@author: Kamil
"""
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import quantum_functions_library as qfl
import numpy as np
import numba
import numpy as np
import scipy.io
import sys, os
import datetime

print(datetime.datetime.now())
### For saving figures and clearing up any file paths
file_path = 'C://Users/Kamil/Dropbox/Masters Project/Write up/Images/'
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
T = 0.25
gamma = 0.25 / 2
init_state_str = 'coherent_state_' + 'q' + str(q_init).replace(".", "") + 'p_init' + str(p_init).replace(".", "")

## Load all of the necessary operators
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

### Can plot the x-operator average to gauge the time at which one wants to see
### the probability distribution
#plt.plot(time_array.tolist()[0], x_operator_time_average.tolist()[0])

### Problems importing coherent states as matrices. Have to use this solution instead
dq = 0.3
dp = 0.3
coherent_states, coherent_states_data = qfl.husimi_coherent_states(dim, mass, omega, hbar, qmin, qmax, dq, pmin, pmax, dp)

time_index = 500
wave_func = wave_evol_matrix[:,time_index]
name_of_state = 'evol_coh_state_time_' + str(time_index)

dq = 0.025
dp = 0.025
prob_phase_space, q_grid, p_grid, weights_coh_states = qfl.quantum_phase_space_prob_distr(wave_func, coherent_states, coherent_states_data, qmin, qmax, dq, pmin, pmax, dp)

#### Plot 1
#
#### Contour plot - short time all points
z = prob_phase_space
q, p = np.mgrid[slice(qmin, qmax + dq, dq), 
                slice(pmin, pmax + dp, dp)]
z_min, z_max = 0, np.abs(prob_phase_space).max()
fig, ax = plt.subplots()
plt.pcolormesh(q, p, z, cmap='RdBu', vmin=z_min, vmax=0.0045)
### set the limits of the plot to the limits of the data
plt.axis([q.min(), q.max(), p.min(), p.max()])
ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
col = plt.colorbar()
col.set_label(r'$|\psi(q,p)|^2$', rotation = 270, labelpad = 20)
plt.xlabel(r'$q$')
plt.ylabel(r'$p$')
fig.tight_layout()
plt.savefig(file_path + 'husimi_ind_path_' + name_of_state + '_t' + str(T).replace(".", "") +  '_g' + str(gamma).replace(".", "") + '.pdf')

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(p, q, prob_phase_space, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

ax.set_xlabel('q')
ax.set_ylabel('p')
ax.yaxis.set_major_locator(plt.MaxNLocator(5))
ax.xaxis.set_major_locator(plt.MaxNLocator(5))
ax.zaxis.set_major_locator(plt.MaxNLocator(4))
# Add a color bar which maps values to colors.
ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
col = fig.colorbar(surf, shrink=0.5, aspect=5)
col.set_label(r'$|\psi(q,p)|^2$', rotation = 270, labelpad = 20)
fig.tight_layout()
plt.savefig(file_path + 'husimi_ind_path_3d' + name_of_state + '_t' + str(T).replace(".", "") +  '_g' + str(gamma).replace(".", "") + '.pdf')
plt.show()

#### More for videos - evaluate probability distribution at given time steps
#
dq = 0.025
dp = 0.025
#### Booster function to hopefully speed up the process - does not work. FIX ME.
@numba.jit
def compute_prob_dist_over_time(wave_evol_matrix, time_array, time_skipper, coherent_states, coherent_states_data, qmin, qmax, dq, pmin, pmax, dp):
    adjusted_time_array = time_array[:,::time_skipper]
    adjusted_wave_evol_matrix = wave_evol_matrix[:, ::time_skipper]
    prob_distr_over_time_phase_space = np.zeros((len(np.arange(qmin, qmax + dq, dq)), 
                                                 len(np.arange(pmin, pmax + dp, dp)), 
                                                 adjusted_time_array.shape[1]), dtype = float)
    for j in range(adjusted_time_array.shape[1]):
        if j%10 == 0:
            print(datetime.datetime.now())
        outputs = qfl.quantum_phase_space_prob_distr(np.matrix(adjusted_wave_evol_matrix[:, j]), 
                                                     coherent_states, 
                                                     coherent_states_data, 
                                                     qmin, qmax, dq, pmin, pmax, dp)
        prob_distr_over_time_phase_space[:, :, j] = outputs[0]
    return adjusted_time_array, prob_distr_over_time_phase_space

time_skipper = 100
adjusted_time_array, prob_distr_over_time_phase_space = compute_prob_dist_over_time(wave_evol_matrix[:, 90000:105000], time_array[:, 90000:105000], time_skipper, coherent_states, coherent_states_data, qmin, qmax, dq, pmin, pmax, dp)        
    
##### Save all of the data
data_output_path = './Data/qm_dynamics_presentation_' + init_state_str + \
                   '_t' + str(T).replace(".", "") + '_g' + str(gamma).replace(".", "") + '.mat'

scipy.io.savemat(data_output_path, 
                 mdict={'time_array': adjusted_time_array,
                        'prob_distr_over_time_phase_space': prob_distr_over_time_phase_space}, 
                 oned_as='row')
matdata = scipy.io.loadmat(data_output_path)
assert np.all(adjusted_time_array == matdata['time_array'])
assert np.all(prob_distr_over_time_phase_space == matdata['prob_distr_over_time_phase_space'])
print("checkpoint8")