# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 11:13:58 2018

@author: Kamil

Script to show the convergence of integrating the Liouville equation for density
matrix explicity and averaging over realisations of the stochastic Schrodinger
equation.
"""
import scipy.io
import numpy as np
import matplotlib.pyplot as plt

#### For saving figures and clearing up any file paths
file_path = './Images/'

T = 0.25
gamma = 0.01/2
# Initial conditions
p_init = 0.1 
q_init = 1
init_state_str = 'coherent_state_' + 'q' + str(q_init).replace(".", "") + 'p_init' + str(p_init).replace(".", "")

##### Data Path
data_output_path_averaging = './Data/qm_sse_simulation_averaging_paths_' + init_state_str + \
                   '_t' + str(T).replace(".", "") + '_g' + str(gamma).replace(".", "") + '.mat'

data_output_path_lindblad = './Data/qm_lindblad_simulation_conv_data' + init_state_str + \
                   '_t' + str(T).replace(".", "") + '_g' + str(gamma).replace(".", "") + '.mat'

try:
    matdata = scipy.io.loadmat(data_output_path_averaging)
    time_array = np.matrix(matdata['time_array'])
    x_operator_time_average = np.matrix(matdata['x_operator_time_average'])
    x_time_standard_error = np.matrix(matdata['x_time_standard_error'])
    p_operator_time_average = np.matrix(matdata['p_operator_time_average'])
    p_time_standard_error = np.matrix(matdata['p_time_standard_error'])
    x_operator_time_paths = np.matrix(matdata['x_operator_time_paths'])
    p_operator_time_paths = np.matrix(matdata['p_operator_time_paths'])
except Exception as e:
    raise('Data has to be generated first')

try:
    matdata = scipy.io.loadmat(data_output_path_lindblad)
    rho_operator_short_time = matdata['rho_operator_short_time']
    x_operator_average = matdata['x_operator_average']
    p_operator_average = matdata['p_operator_average']
except Exception as e:
    raise('Data has to be generated first')

plt.figure()
plt.plot(time_array.tolist()[0], x_operator_average.tolist()[0], label = 'Exact Solution')
plt.plot(time_array.tolist()[0], np.mean(x_operator_time_paths[:, :50], axis = 1), label = 'R=50')
plt.plot(time_array.tolist()[0], np.mean(x_operator_time_paths[:, :100], axis = 1), label = 'R=100')
plt.plot(time_array.tolist()[0], x_operator_time_average.tolist()[0], label = 'R=250')
plt.ylabel(r'$\langle q \rangle$')
plt.xlabel('t')
plt.legend()
x1,x2,y1,y2 = plt.axis()
plt.axis((x1,x2,y1,y2))
plt.show()
plt.savefig(file_path + 'convergence_xoperator_t' + str(T).replace(".", "") +  '_g' + str(gamma).replace(".", "") + '.pdf')

##### MSE  plots
averages_fifty = np.mean(x_operator_time_paths[:, :50], axis = 1)
averages_hundred = np.mean(x_operator_time_paths[:, :100], axis = 1)
averages_all = np.mean(x_operator_time_paths[:, :250], axis = 1)
error_fifty = np.zeros(x_operator_average.shape[1], dtype = float) 
error_hundred = np.zeros(x_operator_average.shape[1], dtype = float) 
error_all = np.zeros(x_operator_average.shape[1], dtype = float)
for j in range(x_operator_average.shape[1]):
    error_fifty[j] = np.abs(averages_fifty[j, 0] - x_operator_average[0, j]) ** 2
    error_hundred[j] = np.abs(averages_hundred[j, 0] - x_operator_average[0, j]) ** 2
    error_all[j] = np.abs(averages_all[j, 0] - x_operator_average[0, j]) ** 2
plt.figure()
plt.plot(time_array.tolist()[0], np.sqrt(error_fifty), label = 'Mean square error for R=50')
plt.plot(time_array.tolist()[0], np.sqrt(error_hundred), label = 'Mean square error for R=100')
plt.plot(time_array.tolist()[0], np.sqrt(error_all), label = 'Mean square error for R=250')
plt.ylabel(r'$\sqrt{|\langle q \rangle_R - \langle q \rangle_{exact}|^2}$')
plt.xlabel('t')
plt.legend()
x1,x2,y1,y2 = plt.axis()
plt.axis((x1,x2,y1,y2))
plt.show()
plt.savefig(file_path + 'convergence_xoperator_mse_t' + str(T).replace(".", "") +  '_g' + str(gamma).replace(".", "") + '.pdf')
print(np.mean(np.sqrt(error_fifty)))
print(np.mean(np.sqrt(error_hundred)))
print(np.mean(np.sqrt(error_all)))

#### P operator
plt.figure()
plt.plot(time_array.tolist()[0], p_operator_average.tolist()[0], label = 'Exact Solution')
plt.plot(time_array.tolist()[0], np.mean(p_operator_time_paths[:, :50], axis = 1), label = 'R=50')
plt.plot(time_array.tolist()[0], np.mean(p_operator_time_paths[:, :100], axis = 1), label = 'R=100')
plt.plot(time_array.tolist()[0], p_operator_time_average.tolist()[0], label = 'R=250')
plt.ylabel(r'$\langle p \rangle$')
plt.xlabel('t')
plt.legend()
x1,x2,y1,y2 = plt.axis()
plt.axis((x1,x2,y1,y2))
plt.show()
plt.savefig(file_path + 'convergence_poperator_t' + str(T).replace(".", "") +  '_g' + str(gamma).replace(".", "") + '.pdf')

##### MSE  plots
averages_fifty = np.mean(p_operator_time_paths[:, :50], axis = 1)
averages_hundred = np.mean(p_operator_time_paths[:, :100], axis = 1)
averages_all = np.mean(p_operator_time_paths[:, :250], axis = 1)
error_fifty = np.zeros(p_operator_average.shape[1], dtype = float) 
error_hundred = np.zeros(p_operator_average.shape[1], dtype = float) 
error_all = np.zeros(p_operator_average.shape[1], dtype = float)
for j in range(p_operator_average.shape[1]):
    error_fifty[j] = np.abs(averages_fifty[j, 0] - p_operator_average[0, j]) ** 2
    error_hundred[j] = np.abs(averages_hundred[j, 0] - p_operator_average[0, j]) ** 2
    error_all[j] = np.abs(averages_all[j, 0] - p_operator_average[0, j]) ** 2
plt.figure()
plt.plot(time_array.tolist()[0], np.sqrt(error_fifty), label = 'Mean square error for R=50')
plt.plot(time_array.tolist()[0], np.sqrt(error_hundred), label = 'Mean square error for R=100')
plt.plot(time_array.tolist()[0], np.sqrt(error_all), label = 'Mean square error for R=250')
plt.ylabel(r'$\sqrt{|\langle p \rangle_R - \langle p \rangle_{exact}|^2}$')
plt.xlabel('t')
plt.legend()
x1,x2,y1,y2 = plt.axis()
plt.axis((x1,x2,y1,y2))
plt.show()
plt.savefig(file_path + 'convergence_poperator_mse_t' + str(T).replace(".", "") +  '_g' + str(gamma).replace(".", "") + '.pdf')
print(np.mean(np.sqrt(error_fifty)))
print(np.mean(np.sqrt(error_hundred)))
print(np.mean(np.sqrt(error_all)))