# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 21:20:27 2018

@author: Kamil

Script for plotting the phase space trajectories without temperature effects.
"""
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np

q = np.arange(-1.5, 1.5, 0.0001)
q = np.concatenate((q, np.arange(-1.7, -1.4, 0.0001), np.arange(1.4, 1.7, 0.0001)))
q = np.sort(q)
E = [-0.5, 0, 0.5]
energy_colors = ['b', 'r', 'g']

def return_trajectories(E, q_list):
    '''Inputs:
       q - as array. The list of q coordinates
       E - as float. The list of energy values
       Outputs:
       trajectories - as list. The list of trajectories. Each entry is [q, p], 
                      where p, q are array of coordinates'''
    trajectories = []
    for j in range(len(E)):
        if E[j] < 0:
            q = q_list[np.where(q_list<0)]
            q_coords = []
            p_coords = []
            for i in range(len(q)):
                p = np.sqrt(E[j] + 1.5 * q[i] ** 2 - 0.75 * q[i] ** 4)
                if not np.isnan(p):
                    p_coords.append(p)
                    q_coords.append(q[i])
            q = q[::-1]
            for i in range(len(q)):
                p = np.sqrt(E[j] + 1.5 * q[i] ** 2 - 0.75 * q[i] ** 4)
                if not np.isnan(p):
                    p_coords.append(-p)
                    q_coords.append(q[i])
            trajectories.append([q_coords, p_coords, E[j]])
            q = q_list[np.where(q_list>0)]
            q_coords = []
            p_coords = []
            for i in range(len(q)):
                p = np.sqrt(E[j] + 1.5 * q[i] ** 2 - 0.75 * q[i] ** 4)
                if not np.isnan(p):
                    p_coords.append(p)
                    q_coords.append(q[i])
            q = q[::-1]
            for i in range(len(q)):
                p = np.sqrt(E[j] + 1.5 * q[i] ** 2 - 0.75 * q[i] ** 4)
                if not np.isnan(p):
                    p_coords.append(-p)
                    q_coords.append(q[i])
            trajectories.append([q_coords, p_coords, E[j]])
        else:
            q_coords = []
            p_coords = []
            q = q_list
            for i in range(len(q)):
                p = np.sqrt(E[j] + 1.5 * q[i] ** 2 - 0.75 * q[i] ** 4)
                if not np.isnan(p):
                    p_coords.append(p)
                    q_coords.append(q[i])
            q = q[::-1]
            for i in range(len(q)):
                p = np.sqrt(E[j] + 1.5 * q[i] ** 2 - 0.75 * q[i] ** 4)
                if not np.isnan(p):
                    p_coords.append(-p)
                    q_coords.append(q[i])
            trajectories.append([q_coords, p_coords, E[j]])
    return trajectories
        
trajectories = return_trajectories(E, q)      
fig, ax = plt.subplots()
E = np.array(E)
for i in range(len(trajectories)):
    index = 0
    for j in range(len(E)):
        if trajectories[i][2] == E[j]:
            index = j
            break
    plt.plot(trajectories[i][0], trajectories[i][1], color = energy_colors[index] , label = 'E=' + str(trajectories[i][2]))

handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc = 'upper left')