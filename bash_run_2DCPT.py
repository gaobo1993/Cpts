# -*- coding: cp936 -*-
from math import *
import visual as vs

import random

import numpy as np
import pylab
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import pyplot as plt


def dist( pos1, pos2 ):
    return sqrt( pow(pos1[0]-pos2[0], 2) + pow(pos1[1]-pos2[1], 2) + pow(pos1[2]-pos2[2], 2))

xRange = 121
yRange = 121
zRange = 1

density = [0 for i in range( xRange*yRange*zRange )]

def getDensity(pos):
    if (xRange-1)/2>= pos[0] >= -(xRange-1)/2 and (yRange-1)/2 >= pos[1] >= -(yRange-1)/2 and zRange-1 >= pos[2] >= 0:
        return density[int(((pos[0]+(xRange-1)/2)*yRange+(pos[1]+(yRange-1)/2))*zRange+pos[2])]
    else:
        return -1


# Set the environment here
def drawEnv( sourcePos = (-40, 0, 0), step = 5):
    # Set the environmen(density)
    for i in range(-int((xRange-1)/2), int((xRange-1)/2)):
        for j in range(-int((yRange-1)/2), int((yRange-1)/2)):
            for k in range (0, int(zRange)):
                num = int(((i+(xRange-1)/2)*yRange + (j+(yRange-1)/2))*zRange + k)
                p = (i, j, k)            
                density[num] = 1 / (dist(sourcePos, p)+0.00001)



def makeHoriVector(theta):
    return vs.vector(cos(theta), sin(theta), 0)

def makeVector(theta, phi):
    return vs.vector(cos(theta)*cos(phi), sin(theta)*cos(phi), sin(phi))


def sigmoid(x, alpha = -4):
    return 1.0 / (1.0+exp(-alpha*x)) if x < 10 else 0


# Implement your strategy for vehicle here
def loopVehicle( vsrcPos = vs.vector(-20, 0, 30), initPos = (40, 40, 0), theta = 3/2*pi, phi = 0, delta = 0.5, W = 4, alpha = -4 ):
    """ Init & loop vehicle.
    """
    
    initAxis = (cos(theta)*cos(phi), sin(theta)*cos(phi), sin(phi))
    
    vPos, vAxis, vVelocity = vs.vector(initPos), vs.vector(initAxis), vs.vector(initAxis)
    deltat = delta
    vscale = 8

    time = 0
    sampleTime = 500
    sampleN = 0
    distence = 0

    while True:
        #vs.rate(3000)

        orthV = makeHoriVector(theta+pi/2)
        orthV2 = makeVector(theta, phi+pi/2)
        lSensorPos = tuple(int(x) for x in (vPos+orthV*W/2).astuple())
        rSensorPos = tuple(int(x) for x in (vPos-orthV*W/2).astuple())
        #print(rSensorPos)

        if (getDensity(lSensorPos) == -1 or getDensity(rSensorPos) == -1) :
            vVelocity = -vVelocity
            vAxis = vVelocity
            vPos = vPos+vVelocity*deltat
            continue
        
        if (getDensity(lSensorPos) > getDensity(rSensorPos)):
            theta = theta + pi/180
        elif getDensity(lSensorPos) < getDensity(rSensorPos):
            theta = theta - pi/180

        vVelocity = makeVector(theta, phi)    
        vPos = vPos+vVelocity*deltat*sigmoid(getDensity(lSensorPos)+getDensity(rSensorPos), alpha)
        vAxis = vVelocity

        if sampleN == 0 and vs.mag(vsrcPos-vPos) > 2:
            time = time + 1
        else:
            sampleN = sampleN+1
            distence = distence+vs.mag(vsrcPos-vPos)
            if sampleN == sampleTime:
                return (time, distence/sampleTime)


drawEnv(sourcePos = (-20, 0, 0))
#print loopVehicle(theta = random.randint(1,4)/2*pi, initPos = (20, 20, 45), alpha = 0, delta = 0.01)

alphaList = [0, -0.5, -1, -2, -3, -4, -5, -8]
#alphaList = [0, -4, -5, -8]
stepList = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1]
#stepList = [0.05, 1]

time = np.zeros((len(stepList), len(alphaList)))

distense = np.zeros((len(stepList), len(alphaList)))

for itr in range(len(stepList)):
    for k in range(len(alphaList)):
        for i in range(1, 4):
            for j in range(1, 3):
                temp = loopVehicle(vsrcPos = vs.vector(-20, 0, 0), theta = i/2*pi, initPos = (20, 20, 0), alpha = alphaList[k], delta = stepList[itr])
                time[itr][k] = time[itr][k]+temp[0]
                distense[itr][k] = distense[itr][k]+temp[1]

time = [x/12 for x in time]
distense = [x/12 for x in distense]

print time, distense


pylab.figure(1)
plot1 = [pylab.plot(alphaList, time[i], 's-', label='Step is '+str(stepList[i])) for i in range(len(stepList))]
pylab.legend(loc=1)
pylab.title('T: Time to get within reasonable distance to source')
pylab.xlabel(r'$\alpha$')
pylab.ylabel('T')
pylab.grid()

pylab.savefig('./figure/time1.png', bbox_inches='tight')


pylab.figure(2)
plot2 = [pylab.plot(alphaList, distense[i], 's-', label='Step is '+str(stepList[i])) for i in range(len(stepList))]
pylab.legend(loc=2)
pylab.title('D: Average distance to source after first locating')
pylab.xlabel(r'$\alpha$')
pylab.ylabel('D')
pylab.grid()

pylab.savefig('./figure/distanse1.png', bbox_inches='tight')


fig = pylab.figure(3)
ax = fig.add_subplot(111, projection='3d')

X, Y = np.meshgrid(alphaList, stepList)
surf=ax.plot_surface(X, Y, time, rstride=1, cstride=1, cmap=cm.coolwarm,
        linewidth=0, antialiased=False)
ax.set_xlabel('Step')
ax.set_ylabel(r'$\alpha$')
ax.set_zlabel('T')
pylab.title('T: Time to get within reasonable distance to source')
fig.colorbar(surf)
ax.grid()

pylab.savefig('./figure/time2.png', bbox_inches='tight')


fig = pylab.figure(4)
ax = fig.add_subplot(111, projection='3d')

X, Y = np.meshgrid(alphaList, stepList)
surf=ax.plot_surface(X, Y, distense, rstride=1, cstride=1, cmap=cm.coolwarm,
        linewidth=0, antialiased=False)
ax.set_xlabel('Step')
ax.set_ylabel(r'$\alpha$')
ax.set_zlabel('D')
pylab.title('D: Average distance to source after first locating')
fig.colorbar(surf)
ax.grid()

pylab.savefig('./figure/distanse2.png', bbox_inches='tight')


pylab.show()

