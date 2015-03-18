from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt

step = 0.04
maxval = 1.0
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# create supporting points in polar coordinates
X = np.arange(-5, 5, 0.25)
Y = np.arange(-10, 10, 0.25)

Z = np.zeros((np.size(X), np.size(Y)))
print np.size(Z[0], 0)
Z1 = [ Z[i] for i in range(np.size(Z, 0)) ]
print np.size(Z1, 0)

X, Y = np.meshgrid(Y,X)

surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
        linewidth=0, antialiased=False)
ax.set_zlim3d(0, 1)
ax.set_xlabel(r'$\phi_\mathrm{real}$')
ax.set_ylabel(r'$\phi_\mathrm{im}$')
ax.set_zlabel(r'$V(\phi)$')
plt.show()
