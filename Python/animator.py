# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 19:52:41 2018

@author: Kamil

Class for Animating numpy array data to create the visualisation of evolution of 
either the particle trajectories or density matrix evolution.
"""

import numpy as np
from collections import defaultdict

### Latest draft. Still needs proper commenting. Basic framework. Further work
### work required.

#### TO DO:
#### Could potentially add annotations to the video. Fix annotations and labelling.
#### Could potentially add plots next to each other, etc.

class Animator():
    """ Animator animates a numpy arrays in one or more subplots, using
    specified arrays and the specified time array.

    Annotations can be added. This can be used to e.g. create case studies or
    explainers.

    Note that the animate() function returned by get_animate_function() must
    be run in the correct order, otherwise the behaviour is undefined.

    """
    def __init__(
            self,
            time_array,
            x,
            plt,
            xlabel,
            ylabel,
            y = None,
            z = None,
            figsize=None,
            title=None,
            use_legends=True,
            legends_loc=None,
            legends_rename=None,
            ylim_min_multiplier_default=None,
            ylim_max_multiplier_default=None,
            speed=1):
        self.time_array = time_array
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.x = x
        self.y = y
        self.z = z
        self.plt = plt
        self.figsize = figsize
        self.use_legends = use_legends
        self.legends_loc = legends_loc
        self.legends_rename = legends_rename
        
        # columns_lhs_per_subplot = list of list of str
        self.DEBUG = True
        self.SPEED = speed
        assert(type(self.SPEED) is int), 'speed needs to be int'
        
        if legends_loc is None:
            self.legends_loc = 'lower left'
        else:
            self.legends_loc = legends_loc
        
        self.plt = plt  # the pyplot object
        
        self.fig = self.plt.figure(figsize=figsize)
        self.ax = self.plt.axes()
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)
        
        if self.y is None and self.z is None:
            self.type_of_graph = 'grow_line'
            assert(len(self.time_array) == len(self.x))
            xmin = np.min(self.time_array)
            xmax = np.max(self.time_array)
            ymin = np.min(self.x) * 0.95
            ymax = np.max(self.x) * 1.15
            self.ax.set_xlim([xmin,xmax])
            self.ax.set_ylim([ymin,ymax])
        elif self.y is not None and self.z is None:
            self.type_of_graph = 'line'
            assert(len(self.time_array) == len(self.y))
            ### FIX ME. Current way of finding min and max values might be problematic
            ### in the long run. For current purposes works fine.
            xmin = np.min(self.x)
            xmax = np.max(self.x)
            ymin = 10000
            ymax = -10000
            for j in range(len(self.y)):
                if np.min(self.y[j]) < ymin:
                    ymin = np.min(self.y[j])
                if np.max(self.y[j]) > ymax:
                    ymax = np.max(self.y[j]) 
            self.ax.set_xlim([xmin,xmax])
            self.ax.set_ylim([ymin * 0.90 , ymax * 1.15])
        elif self.y is not None and self.z is not None:
            self.type_of_graph = 'contour'
            assert(len(self.x) == self.z.shape[0])
            assert(len(self.y) == self.z.shape[1])
            
        self.contours = []
        self.lines = []

    def get_number_of_frames_required(self):
        return len(self.time_array)

    def get_init_function(self):
        def init_animate():
            if self.type_of_graph == 'line' or 'grow_line':
                # initialise all lines
                l, = self.ax.plot([], [], lw=1)
                l.set_data([], [])
                self.lines.append(l)
            elif self.type_of_graph == 'contour':
                x, y = np.meshgrid(self.x, self.y)
                z = self.z[:,:, 0]
                cont, = self.ax.contourf(x, y, z, 500)
                self.contours.append(cont)
            return (*self.lines, *self.contours)
        return init_animate

    def get_animate_function(self):
        def animate(i):
            # i is the frame number
            for line_idx, l in enumerate(self.lines):
                # line_idx = line_idx of the line
                # plot all lines
                if self.type_of_graph == 'grow_line':
                    x = self.time_array[0:i]
                    y = self.y[0:i]
                    l.set_data(x, y)
                if self.type_of_graph == 'line':
                    x = self.x
                    y = self.y[i]
                    l.set_data(x, y)
            
            for contour_idx, cont in enumerate(self.contours):
                # line_idx = line_idx of the line
                # plot all lines
                if self.type_of_graph == 'contour':
                    x = self.x
                    y = self.y
                    x, y = np.meshgrid(x,y)
                    z = self.z[:,:,i]
                    cont.set_data(x,y,z)
            return (*self.lines, *self.contours)
        return animate