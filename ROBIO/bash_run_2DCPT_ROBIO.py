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
def loopVehicle( vsrcPos = vs.vector(-20, 0, 30), initPos = (40, 40, 0), theta = 3/2*pi, phi = 0, delta = 0.5, W = 4, alpha = -4, thre_den = 0.4 ):
    """ Init & loop vehicle.
    """
    
    initAxis = (cos(theta)*cos(phi), sin(theta)*cos(phi), sin(phi))
    
    vPos, vAxis, vVelocity = vs.vector(initPos), vs.vector(initAxis), vs.vector(initAxis)
    deltat = delta
    vscale = 8

    time = 0
    sampleTime = 500
    sampleN = 0
    distance = 0

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

        if (getDensity(lSensorPos)+getDensity(rSensorPos))/2 <= thre_den:
            time = time + 1
        else:
            distance = distance+vs.mag(vsrcPos-vPos)
            return (time, distance)
        if time >= sampleTime:
            return (time, distance)


drawEnv(sourcePos = (-20, 0, 0))

thre_den = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]

time = [0 for i in range(len(thre_den))]

distance = [0 for i in range(len(thre_den))]

for k in range(len(thre_den)):
    for i in range(1, 4):
        temp = loopVehicle(vsrcPos = vs.vector(-20, 0, 0), theta = i/2*pi, initPos = (20, 20, 0), alpha = 0, delta = 1, thre_den = thre_den[k])
        time[k] = time[k]+temp[0]
        distance[k] = distance[k]+temp[1]
            
time = [x/4 for x in time]
distance = [x/4 for x in distance]


fig1 = plt.figure(1)
ax1 = fig1.add_subplot(111)
plotl = ax1.plot(thre_den, time, '-b', label='Time')
ax1.set_xlabel('Threshold of source declaration')
ax1.set_ylabel('Time', color='b')
ax1.set_ylim(115, 120)
for tl in ax1.get_yticklabels():
    tl.set_color('b')
ax1.legend(loc=1)

ax2 = ax1.twinx()
plotr, = ax2.plot(thre_den, distance, '--ro', label='distance')
ax2.set_ylabel('distance', color='r')
for tl in ax2.get_yticklabels():
    tl.set_color('r')
ax2.legend(loc = 2)
plt.grid()
plt.savefig('./ROBIO/2d_t_d.png', bbox_inches='tight')

plt.show()

