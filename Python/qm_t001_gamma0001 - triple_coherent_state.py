# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 12:29:44 2018

@author: Kamil

Script for integrating Lindblad equation directly for various parameters of model
and different initial conditions. Output are figures for write-up.
"""

from animator import *
from quantum_functions_library import *
import numpy as np
import matplotlib.pyplot as plt
import numba
import timeit
from matplotlib import animation
import numpy as np
import scipy.io
import sys, os

### For saving figures and clearing up any file paths
file_path = '../Write up/Images/'
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

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

### Defining operators independent of T and gamma.
x_operator = np.sqrt(hbar / (2 * mass * omega)) * (raising_operator(dim) + lowering_operator(dim)) 
p_operator = 1j * np.sqrt(hbar * mass * omega / 2) * (raising_operator(dim) - lowering_operator(dim)) 
h_sys = 1/(2 * mass) * p_operator * p_operator + V2 * x_operator * x_operator * x_operator * x_operator - V1 * x_operator * x_operator

##### Have checkpoints to debug code. Also allow to see which calculations have already been carried out
print("checkpoint1")

project_position_x0 = projection_position(dim, 0, mass, omega, hbar)
current_density_flux_at_x0 = (0.5 * mass) * (project_position_x0 * p_operator + p_operator.T * project_position_x0)

#### Load all of the necessary operators
try:
    data_path = './Data/qm_tools_dim_' + str(dim).replace(".", "") + '_mass_' + str(mass).replace(".", "") + '_omega_' + \
                str(omega).replace(".", "") + '.mat'
    matdata = scipy.io.loadmat(data_path)
    project_A = np.matrix(matdata['project_A'])
    x_grid_projectors = matdata['x_grid_projectors']
    for j in range(len(x_grid_projectors)):
        x_grid_projectors[j] = np.matrix(x_grid_projectors[j])
    x_grid_midpoints = matdata['x_grid_midpoints']
    xmin = matdata['x_limits'][0][0]
    xmax = matdata['x_limits'][0][1]
except Exception as e:
    raise('Data has to be generated first')

### Defining constants and operators that will be adjusted in project
gamma = 0.001
T = 0.01
lindblad = np.sqrt(2 * mass * kb * T) / hbar * x_operator + 1j / (2 * np.sqrt(2 * mass * kb * T)) * p_operator
hamiltonian = h_sys + gamma/2 * (x_operator * p_operator + p_operator * x_operator)

### Triple coherent state for this simulation
p_init0 = np.sqrt(3 * kb * T * mass)/80
q_init0 = -1.25
p_init1 = np.sqrt(3 * kb * T * mass)/80
q_init1 = -2.25
p_init2 = np.sqrt(3 * kb * T * mass)/80
q_init2 = -0.25

#wavefunc_state1 = init_coherent_state(dim, p_init0, q_init0, mass, omega, hbar)
#init_rho_state1 = wavefunc_state1 * wavefunc_state1.T
#init_rho_state1 = init_rho_state1 / trace_matrix(init_rho_state1)
#
#wavefunc_state2 = init_coherent_state(dim, p_init1, q_init1, mass, omega, hbar)
#init_rho_state2 = wavefunc_state2 * wavefunc_state2.T
#init_rho_state2 = init_rho_state2 / trace_matrix(init_rho_state2)
#
#wavefunc_state3 = init_coherent_state(dim, p_init2, q_init2, mass, omega, hbar)
#init_rho_state3 = wavefunc_state3 * wavefunc_state3.T
#init_rho_state3 = init_rho_state3 / trace_matrix(init_rho_state3)

wavefunc = 1/np.sqrt(3) * init_coherent_state(dim, p_init0, q_init0, mass, omega, hbar) + \
           + 1/np.sqrt(3) * init_coherent_state(dim, p_init1, q_init1, mass, omega, hbar) + \
           + 1/np.sqrt(3) * init_coherent_state(dim, p_init2, q_init2, mass, omega, hbar)
init_rho = wavefunc * wavefunc.T
#init_rho = init_rho / trace_matrix(init_rho)

## Define name to save figures automatically
init_state_str = 'triple_coherent_state_'

### Plotting the initial state

### Creating quicker function to plot the initial probability density distribution.
### Have not timed it. Effectiveness is dubious. Should probably consider vectorising
### but that would probably involve re-writing trace_matrix to accomodate arrays
### or finding apply function. FIX ME.

@numba.jit
def init_prob_density_distr(init_rho, x_grid_projectors):
    init_prob_den_distr = np.zeros(len(x_grid_projectors), dtype = float)
    for k in range(len(x_grid_projectors)):
        init_prob_den_distr[k] = trace_matrix(x_grid_projectors[k] * init_rho)
    return init_prob_den_distr

#init_prob_den_distr1 = init_prob_density_distr(init_rho_state1, x_grid_projectors)
#init_prob_den_distr2 = init_prob_density_distr(init_rho_state2, x_grid_projectors)
#init_prob_den_distr3 = init_prob_density_distr(init_rho_state3, x_grid_projectors)

init_prob_den_distr = init_prob_density_distr(init_rho, x_grid_projectors)

### Actual plot
plt.figure()
#plt.plot(x_grid_midpoints[0], init_prob_den_distr1)
#plt.plot(x_grid_midpoints[0], init_prob_den_distr2)
#plt.plot(x_grid_midpoints[0], init_prob_den_distr3)
plt.plot(x_grid_midpoints[0], init_prob_den_distr)
plt.plot()
plt.ylabel(r'$Tr(\rho(q))$')
plt.xlabel(r'$q$')
x1,x2,y1,y2 = plt.axis()
plt.axis((xmin,xmax,0,0.05))
plt.show()
plt.savefig(file_path + 'qm_init_dist_dm' + init_state_str)

print("checkpoint4")

#### RESULTS START HERE

#### Getting the time evolution of rho and prob density

### For most of graphs
short_time_step = 0.1
short_no_steps = 500

### To show actual transition rate goes to zero
long_time_step = 1
long_no_steps = 1000

rho_operator_short_time = rho_lindblad_superoperator(init_rho, hamiltonian, lindblad, hbar, short_time_step, short_no_steps)
rho_operator_long_time = rho_lindblad_superoperator(init_rho, hamiltonian, lindblad, hbar, long_time_step, long_no_steps)

print("checkpoint5")

short_time = np.arange(0, short_no_steps * short_time_step, short_time_step)
long_time = np.arange(0, long_no_steps * long_time_step, long_time_step)

@numba.jit
def prob_and_proj_time(rho_operator, time):
    prob_in_time = np.empty((len(time), len(x_grid_midpoints[0])), dtype = float)
    proj_A_time = np.empty(len(time), dtype = float)
    for j in range(len(time)):
        rho = np.matrix(rho_operator[:, :, j])
        proj_A_time[j] = trace_matrix(rho * project_A)       
        for k in range(len(x_grid_projectors)):
            prob_in_time[j, k] = trace_matrix(x_grid_projectors[k] * rho)
    return prob_in_time, proj_A_time

prob_in_short_time, proj_A_short_time = prob_and_proj_time(rho_operator_short_time, short_time)
prob_in_long_time, proj_A_long_time = prob_and_proj_time(rho_operator_long_time, long_time)
    
print("checkpoint6")

short_tran_rate = np.log(4 * np.abs((proj_A_short_time[1:] - 0.5)))/(-2 * short_time[1:])
print(short_tran_rate[-1])
long_tran_rate = np.log(4 * np.abs((proj_A_long_time[1:] - 0.5)))/(-2 * long_time[1:])
print(long_tran_rate[-1])

### Plotting expectation value of being in P_A
plt.figure()
plt.plot(short_time, proj_A_short_time)
plt.ylabel(r'$\langle P_A(t) \rangle$')
plt.xlabel('t')
plt.savefig(file_path + 'qm_proj_A_t' + str(T).replace(".", "") + '_gamma' + \
            str(gamma).replace(".", "") + '_' + init_state_str + 'long_time' + '.png')

plt.figure()
plt.plot(long_time, proj_A_long_time)
plt.ylabel(r'$\langle P_A(t) \rangle$')
plt.xlabel('t')
plt.savefig(file_path + 'qm_proj_A_t' + str(T).replace(".", "") + '_gamma' + \
            str(gamma).replace(".", "") + '_' + init_state_str + 'limit_time' + '.png')

### To highlight behaviour for shorter time periods
plt.figure()
plt.plot(short_time[0:100], proj_A_short_time[0:100])
plt.ylabel(r'$\langle P_A(t) \rangle$')
plt.xlabel('t')
plt.savefig(file_path + 'qm_proj_A_t' + str(T).replace(".", "") + '_gamma' + \
            str(gamma).replace(".", "") + '_' + init_state_str + 'short_time' + '.png')

plt.figure()
plt.plot(short_time[0:100], np.log(4 * np.abs((proj_A_short_time[0:100] - 0.5))))
plt.ylabel(r'$ln(\frac{\langle P_A(t) \rangle - 0.5}{0.25})$')
plt.xlabel('t')
plt.savefig(file_path + 'qm_log_A_t' + str(T).replace(".", "") + '_gamma' + \
            str(gamma).replace(".", "") + '_' + init_state_str + 'short_time' + '.png')

plt.figure()
plt.plot(short_time, np.log(4 * np.abs((proj_A_short_time - 0.5))))
plt.ylabel(r'$ln(\frac{\langle P_A(t) \rangle - 0.5}{0.25})$')
plt.xlabel('t')
plt.savefig(file_path + 'qm_log_A_t' + str(T).replace(".", "") + '_gamma' + \
            str(gamma).replace(".", "") + '_' + init_state_str + 'long_time' + '.png')

### Have to cut off some initial points as original transition rate is quite high and
### dissolves the resolution for behaviour around plateau
plt.figure()
plt.plot(short_time[10:], short_tran_rate[9:])
plt.ylabel(r'$k^q(t)$')
plt.xlabel('t')
plt.savefig(file_path + 'qm_tran_rate_t' + str(T).replace(".", "") + '_gamma' + \
            str(gamma).replace(".", "") + '_' + init_state_str + 'short_time' + '.png')

### Showing that the transition rate goes to zero for long time periods
plt.figure()
plt.plot(long_time[1:], long_tran_rate)
plt.ylabel(r'$k^q(t)$')
plt.xlabel('t')
plt.savefig(file_path + 'qm_tran_rate_t' + str(T).replace(".", "") + '_gamma' + \
            str(gamma).replace(".", "") + '_' + init_state_str + 'long_time' + '.png')

### Contour plot - short time all points
dx = x_grid_midpoints[0][1] - x_grid_midpoints[0][0]
z = prob_in_short_time
y, x = np.mgrid[slice(0, short_no_steps * short_time_step, short_time_step), 
                slice(xmin, xmax, dx)]
z_min, z_max = 0, np.abs(prob_in_short_time).max()
plt.figure()
plt.pcolormesh(x, y, z, cmap='RdBu', vmin=z_min, vmax=z_max)
### set the limits of the plot to the limits of the data
plt.axis([x.min(), x.max(), y.min(), y.max()])
plt.colorbar()
plt.xlabel(r'$q$')
plt.ylabel(r'$t$')
plt.savefig(file_path + 'qm_contour_dm_evol_t' + str(T).replace(".", "") + '_gamma' + \
            str(gamma).replace(".", "") + '_' + init_state_str + 'long_time' + '.png')

### Contour plot - short time first 10% of the points. Need integer type to avoid
### type errors in slice.
z = prob_in_short_time[0:50]
y, x = np.mgrid[slice(0, int(short_no_steps * 0.1) * short_time_step, short_time_step), 
                slice(xmin, xmax, dx)]
plt.figure()
### Use previous zmin and zmax to be able to compare the two plots.
plt.pcolormesh(x, y, z, cmap='RdBu', vmin=z_min, vmax=z_max)
### set the limits of the plot to the limits of the data
plt.axis([x.min(), x.max(), y.min(), y.max()])
plt.colorbar()
plt.xlabel(r'$q$')
plt.ylabel(r'$t$')
plt.savefig(file_path + 'qm_contour_dm_evol_t' + str(T).replace(".", "") + '_gamma' + \
            str(gamma).replace(".", "") + '_' + init_state_str + 'short_time' + '.png')

print("checkpoint7")

### Transition rate using current density formulation

@numba.jit
def eval_current_density_tran_rate(rho_operator, time):
    current_density_tran_rate = np.empty(len(time), dtype = float)
    for j in range(len(time)):
        rho = np.matrix(rho_operator[:, :, j])
        current_density_tran_rate[j] = trace_matrix(current_density_flux_at_x0 * project_A * rho)/0.5
    return current_density_tran_rate

short_tran_rate_current_density = eval_current_density_tran_rate(rho_operator_short_time, short_time)
long_tran_rate_current_density = eval_current_density_tran_rate(rho_operator_long_time, long_time)
print(short_tran_rate_current_density[-1])
print(long_tran_rate_current_density[-1])
print("checkpoint8")

plt.figure()
plt.plot(short_time, short_tran_rate_current_density)
plt.ylabel(r'$k^q(t)$')
plt.xlabel('t')
plt.savefig(file_path + 'qm_curr_den_tran_rate_t' + str(T).replace(".", "") + '_gamma' + \
            str(gamma).replace(".", "") + '_' + init_state_str + 'short_time' + '.png')

plt.figure()
plt.plot(long_time, long_tran_rate_current_density)
plt.ylabel(r'$k^q(t)$')
plt.xlabel('t')
plt.savefig(file_path + 'qm_curr_den_tran_rate_t' + str(T).replace(".", "") + '_gamma' + \
            str(gamma).replace(".", "") + '_' + init_state_str + 'long_time' + '.png')

### Animation maker.

### Very quick way of doing it. Slightly lazy in terms of efficiency. FIX ME.
ydata = []
for j in range(short_no_steps):
    ydata.append(prob_in_short_time[j, :])

df_anim = Animator(
    time_array = short_time[:500],
    x = x_grid_midpoints[0],
    plt = plt,
    xlabel = "q",
    ylabel = r'$Tr(\rho(q))$',
    y = ydata,
    speed=1
)

initf = df_anim.get_init_function()
animf = df_anim.get_animate_function()

nbr_frames = df_anim.get_number_of_frames_required()

#### Animate. blit=True means only re-draw the parts that have changed.

anim = animation.FuncAnimation(df_anim.fig, animf, init_func=initf,
                               frames=nbr_frames,
                               interval=300, blit=True)
print("saving animation...")
anim.save(
    '../Animation/' + 't' + str(T).replace(".", "") + 'gamma' + str(gamma).replace(".", "") + init_state_str + '.mp4',
    fps=10,
    extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])

##### Save all of the data
data_output_path = './Data/qm_lindblad_simulation_' + init_state_str + \
                   '_t' + str(T).replace(".", "") + '_g' + str(gamma).replace(".", "") + '.mat'

scipy.io.savemat(data_output_path, 
                 mdict={'lindblad': lindblad,
                        'hamiltonian': hamiltonian,
                        'init_rho': init_rho,
                        'rho_operator_short_time': rho_operator_short_time,
                        'rho_operator_long_time': rho_operator_long_time,
                        'short_time': short_time,
                        'long_time': long_time,
                        'prob_in_short_time': prob_in_short_time,
                        'proj_A_short_time': proj_A_short_time,
                        'prob_in_long_time': prob_in_long_time,
                        'proj_A_long_time': proj_A_long_time,
                        'short_tran_rate': short_tran_rate,
                        'long_tran_rate': long_tran_rate,
                        'short_tran_rate_current_density': short_tran_rate_current_density,
                        'long_tran_rate_current_density': long_tran_rate_current_density,
                        }, 
                 oned_as='row')
matdata = scipy.io.loadmat(data_output_path)
assert np.all(rho_operator_short_time == matdata['rho_operator_short_time'])
assert np.all(rho_operator_long_time == matdata['rho_operator_long_time'])
assert np.all(short_tran_rate_current_density == matdata['short_tran_rate_current_density'])
print("checkpoint8")