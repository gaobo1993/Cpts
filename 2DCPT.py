from __future__ import division, print_function
import visual as vs   # for 3D panel 
import wx   # for widgets
from math import *

import random

# Draw window & 3D pane =================================================

win = vs.window(width=1024, height=720, menus=False, title='SIMULATE VPYTHON GUI')
                         # make a main window. Also sets w.panel to addr of wx window object. 
#scene = vs.display( window=win, width=830, height=690, forward=-vs.vector(1,1,2))
scene = vs.display( window=win, width=1010, height=690, forward=-vs.vector(1,1,2), background = vs.color.white)
                         # make a 3D panel 
clr = vs.color
vss = scene

# Draw 3D model ======================

def axes( frame, colour, sz, posn ): # Make axes visible (of world or frame).
                                     # Use None for world.   
    directions = [vs.vector(sz,0,0), vs.vector(0,sz,0), vs.vector(0,0,sz)]
    texts = ["X","Y","Z"]
    posn = vs.vector(posn)
    for i in range (3): # EACH DIRECTION
       vs.curve( frame = frame, color = colour, pos= [ posn, posn+directions[i]])
       vs.label( frame = frame,color = colour,  text = texts[i], pos = posn+ directions[i],
                                                                    opacity = 0, box = False )

axes( None, clr.black, 30, (-70,-20,0))


def drawGrid( posn=(0,0,0), sq=10, H=5, W = 8, normal='z', colour= clr.white ) :
    """ Draw grid of squares in XY, XZ or YZ plane with corner nearest origin at given posn.
    sq= length of side of square.  H = number of squares high (Y). W = number of squares wide (X).
    normal is the axis which is normal to the grid plane. 
    """
    ht = H*sq
    wd = W*sq
    for i in range( 0, wd + 1, sq ):  # FOR EACH VERTICAL LINE
        if   normal == 'z':   vs.curve( pos=[(posn[0]+i, posn[1]+ht, posn[2]),
                                             (posn[0]+i, posn[1],    posn[2])], color=colour )
        elif normal == 'x':   vs.curve( pos=[(posn[0], posn[1]+ht,   posn[2]+i),
                                             (posn[0], posn[1],      posn[2]+i)], color=colour)
        else:                 vs.curve( pos=[(posn[0]+i, posn[1], posn[2]+ht),
                                             (posn[0]+i, posn[1], posn[2])], color=colour)
    for i in range( 0, ht+1, sq ):  # FOR EACH HORIZONTAL LINE
        if normal == 'z':   vs.curve( pos=[(posn[0],    posn[1]+i, posn[2]),
                                           (posn[0]+wd, posn[1]+i, posn[2])], color=colour)
        elif normal == 'x': vs.curve( pos=[(posn[0], posn[1]+i, posn[2]+wd),
                                           (posn[0], posn[1]+i, posn[2])], color=colour)
        else:               vs.curve( pos=[(posn[0],    posn[1], posn[2]+i),
                                           (posn[0]+wd, posn[1], posn[2]+i)], color=colour)

drawGrid( normal = 'y', posn= (-60, -60, 0), colour = clr.blue,   W = 12, H = 6 )
#drawGrid( normal = 'z', posn= (-60, 0,  60), colour = clr.blue,   W = 5 )
#drawGrid( normal = 'z', posn= ( 10, 0,  60), colour = clr.blue,   W = 5 )
#drawGrid( normal = 'y', posn = (-60, 60, 0), colour = clr.blue, W = 12 )
drawGrid( normal = 'x', posn= (-60, -60, 0), colour = clr.green,  H = 12, W = 6 )
#drawGrid( normal = 'x', posn= ( 60, -60, 0), colour = clr.green,  H = 12, W = 6 )
drawGrid( normal = 'z', posn= (-60, -60, 0), colour = clr.orange, W = 12, H = 12 )
#drawGrid( normal = 'z', posn= (-60, 0,  0), colour = clr.red,    W = 12 )


def dist( pos1, pos2 ):
    return sqrt( pow(pos1[0]-pos2[0], 2) + pow(pos1[1]-pos2[1], 2) + pow(pos1[2]-pos2[2], 2))

xRange = 121
yRange = 121
zRange = 1

density = [0 for i in range( xRange*yRange*zRange )]

def getDensity(pos):
    if pos[0]+(xRange-1)/2 >= 0 and pos[1]+(yRange-1)/2) >= 0:
        return density[int(((pos[0]+(xRange-1)/2)*yRange+(pos[1]+(yRange-1)/2))*zRange+pos[2])]
    else:
        return -1


# Set the environment here
def drawEnv( sourcePos = (-40, 0, 0), step = 5 ):
    # Set the environmen(density)
    for i in range(-int((xRange-1)/2), int((xRange-1)/2)):
        for j in range(-int((yRange-1)/2), int((yRange-1)/2)):
            for k in range (0, int(zRange)):
                num = int(((i+(xRange-1)/2)*yRange + (j+(yRange-1)/2))*zRange + k)
                p = (i, j, k)            
                density[num] = 1 / (dist(sourcePos, p)+0.00001)

    # Visualize the environmen(density)
    vs.sphere(pos = sourcePos, radius = 2, color = clr.red)
    for i in range(-int((xRange-1)/2), int((xRange-1)/2), step):
        for j in range(-int((yRange-1)/2), int((yRange-1)/2), step):
            for k in range (0, int(zRange), step):
                num = int(((i+(xRange-1)/2)*yRange + (j+(yRange-1)/2))*zRange + k)
                p = (i, j, k) 
                if density[num] < 1:
                    vs.sphere(pos = p, radius = 0.5, color = clr.red, opacity = 3*density[num])
                #else:
                #    vs.sphere(pos = p, radius = 1, color = clr.red)


drawEnv(sourcePos = (-40, 0, 0))

def makeHoriVector(theta):
    return vs.vector(cos(theta), sin(theta), 0)

def makeVector(theta, phi):
    return vs.vector(cos(theta)*cos(phi), sin(theta)*cos(phi), sin(phi))


def sigmoid(x, alpha = -4):
    return 1.0 / (1.0+exp(-alpha*x)) if -alpha*x < 10 else 0


# Implement your strategy for vehicle here
def loopVehicle( initPos = (40, 40, 0), theta = 3/2*pi, phi = 0, delta = 0.5, W = 4 ):
    """ Init & loop vehicle.
    """
    
    initAxis = (cos(theta)*cos(phi), sin(theta)*cos(phi), sin(phi))
    vehicle = vs.box(pos=initPos, size=(W,W,0.2), color = clr.green, axis = initAxis,
                     make_trail=True)
    vehicle.trail_object.radius = 0.2
    vehicle.velocity = vs.vector(initAxis)
    deltat = delta
    vscale = 8
    varr = vs.arrow(pos=vehicle.pos, axis=vscale*vehicle.velocity, color=clr.yellow)

    

    while True:
        vs.rate(1000)

        orthV = makeHoriVector(theta+pi/2)
        orthV2 = makeVector(theta, phi+pi/2)
        lSensorPos = tuple(int(x) for x in (vehicle.pos+orthV*W/2).astuple())
        rSensorPos = tuple(int(x) for x in (vehicle.pos-orthV*W/2).astuple())
        #print(rSensorPos)

        if (getDensity(lSensorPos) > getDensity(rSensorPos)):
            theta = theta + pi/180
        elif getDensity(lSensorPos) < getDensity(rSensorPos):
            theta = theta - pi/180

        vehicle.velocity = makeVector(theta, phi)
    
        vehicle.pos = vehicle.pos+vehicle.velocity*deltat*sigmoid(getDensity(lSensorPos)+getDensity(rSensorPos), -10)
        vehicle.axis = vehicle.velocity
        vehicle.size=(W,W,0.2)

        varr.pos = vehicle.pos
        varr.axis = vehicle.velocity*vscale

        

loopVehicle(theta = random.randint(1,4)/2*pi, initPos = (20, 20, 0))
