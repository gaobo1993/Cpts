from __future__ import division, print_function
import visual as vs   # for 3D panel 
import wx   # for widgets
from math import *

import random

import numpy as np
import pylab
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import pyplot as plt


maxIter = 10
print(range(0, maxIter))

plot1, = pylab.plot(time, y1, 'r')
pylab.show()
