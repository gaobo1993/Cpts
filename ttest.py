from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
import pylab

fig = pylab.figure(1)
ax = fig.add_subplot(111, projection='3d')

alphaList = [0, -0.5, -1, -2, -3, -4, -5, -8]
stepList = [0.05, 0.1, 1]

time = np.random.rand(len(stepList), len(alphaList))


X, Y = np.meshgrid(alphaList, stepList)

surf = ax.plot_surface(X, Y, time, rstride=1, cstride=1, cmap=cm.coolwarm,
        linewidth=0, antialiased=False)
ax.set_zlim3d(0, 1)
ax.set_xlabel(r'$\phi_\mathrm{real}$')
ax.set_ylabel(r'$\phi_\mathrm{im}$')
ax.set_zlabel(r'$V(\phi)$')
ax.grid()

fig.colorbar(surf)

pylab.show()

