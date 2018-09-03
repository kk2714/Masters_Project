# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 14:45:42 2018

@author: Kamil
"""

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np


fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data.
sz = np.arange(-0.5, 0.5, 0.001)
sy = np.arange(-0.5, 0.5, 0.001)
sz, sy = np.meshgrid(sz, sy)
sx_squared = 0.25 * (1 - 2 * sz) * (1 + 2 * sz) ** 2 - sy ** 2
sx = np.sqrt(sx_squared)
sx1 = -np.sqrt(sx_squared)

sy = np.concatenate((sy, sy))
sz = np.concatenate((sz, sz))
sx = np.concatenate((sx1, sx))

# Plot the surface.
surf = ax.plot_surface(sx, sy, sz, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

## Customize the z axis.
#ax.set_zlim(-1.01, 1.01)
#ax.zaxis.set_major_locator(LinearLocator(10))
#ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
#fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()