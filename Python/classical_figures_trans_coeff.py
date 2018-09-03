# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 17:46:08 2018

@author: Kamil

Comments: Script for creating classical transition rate plots.
"""

from classical_functions_library import *
import matplotlib.pyplot as plt

### Note could automate this more by defining parameters at the top, but every
### figure had to be adjusted manually so did not make much sense! Potential FIX
### ME for further automation.
    
# Figure 2.4 page 27
fig_24a = transition_rate(50000, 0.01, 0.001, 0.001, 50)
plt.figure()
plt.plot(fig_24a[0], fig_24a[1])
plt.ylabel(r'$\kappa_{50000}^c$')
plt.xlabel('t')
x1,x2,y1,y2 = plt.axis()
plt.axis((x1,x2,-0.5,1.1))
plt.show()

fig_24b = transition_rate(50000, 0.01, 0.1, 0.001, 50)
plt.figure()
plt.plot(fig_24b[0][1:], fig_24b[1][1:])
plt.ylabel(r'$\kappa_{50000}^c$')
plt.xlabel('t')
x1,x2,y1,y2 = plt.axis()
plt.axis((x1,x2,-0.5,1.1))
plt.show()

fig_24c = transition_rate(50000, 0.01, 0.5, 0.001, 50)
plt.figure()
plt.plot(fig_24c[0][1:], fig_24c[1][1:])
plt.ylabel(r'$\kappa_{50000}^c$')
plt.xlabel('t')
x1,x2,y1,y2 = plt.axis()
plt.axis((x1,x2,-0.5,1.1))
plt.show()

# Figure 2.13a page 35
fig_213a = transition_rate(100000, 0.25, 0.001, 0.001, 50)
plt.figure()
plt.plot(fig_213a[0], fig_213a[1])
plt.ylabel(r'$\kappa_{100000}^c$')
plt.xlabel('t')
x1,x2,y1,y2 = plt.axis()
plt.axis((x1,x2,-1.0,1.1))
plt.show()