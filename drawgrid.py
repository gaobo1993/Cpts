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

win = vs.window(width=1024, height=720, menus=False, title='SIMULATE VPYTHON GUI')
                         # make a main window. Also sets w.panel to addr of wx window object. 
#scene = vs.display( window=win, width=830, height=690, forward=-vs.vector(1,1,2))
scene = vs.display( window=win, width=1010, height=690, forward=-vs.vector(1,1,2), background = vs.color.white)
                         # make a 3D panel 
clr = vs.color
vss = scene

def makeHoriVector(theta):
    return vs.vector(sin(theta), 0, cos(theta))

def makeVector(theta, phi):
    return vs.vector(sin(theta)*cos(phi), sin(phi), cos(theta)*cos(phi))

def axes( frame, colour, sz, posn ): # Make axes visible (of world or frame).
                                     # Use None for world.   
    directions = [vs.vector(sz,0,0), vs.vector(0,sz,0), vs.vector(0,0,sz)]
    texts = ["Y","Z","X"]
    posn = vs.vector(posn)
    for i in range (3): # EACH DIRECTION
       vs.curve( frame = frame, color = colour, pos= [ posn, posn+directions[i]])
       vs.label( frame = frame,color = colour,  text = texts[i], pos = posn+ directions[i],
                                                                    opacity = 0, box = False )

opos = (-20,0,0)
axes( None, clr.black, 15, opos)

W = 10
theta = 1/4*pi
phi = 0
vscale = 1

initAxis = (sin(theta)*cos(phi), sin(phi), cos(theta)*cos(phi))

vinitAxis = vs.vector(initAxis)
orthV = makeVector(theta+pi/2, 0)
orthV2 = makeVector(theta, phi+pi/2)

zxyAxis = (opos[0]+W*sin(theta)*cos(phi), opos[1]+W*sin(phi), opos[2]+W*cos(theta)*cos(phi))
xyAxis = (opos[0]+W*sin(theta)*cos(phi), opos[1], opos[2]+W*cos(theta)*cos(phi))
vehicle = vs.box(pos=opos, size=(W,0.2,W), color = clr.green, axis = initAxis, opacity = 0.5, up = orthV2.astuple())

vs.curve( pos=[opos, xyAxis], color=clr.black)
vs.curve(pos = [xyAxis, zxyAxis], color = clr.black)
vs.curve(pos = [opos, zxyAxis], color = clr.black)

vs.sphere(pos = (vehicle.pos+orthV*W/2).astuple(), radius = 0.5, color = clr.red)
vs.sphere(pos = (vehicle.pos-orthV*W/2).astuple(), radius = 0.5, color = clr.red)

#vs.curve(pos = [(vehicle.pos+orthV2*W/2).astuple(), (vehicle.pos-orthV2*W/2).astuple()], color = clr.blue)
#vs.sphere(pos = (vehicle.pos+orthV2*W/2).astuple(), radius = 0.5, color = clr.red)
#vs.sphere(pos = (vehicle.pos-orthV2*W/2).astuple(), radius = 0.5, color = clr.red)

varr = vs.arrow(pos=vehicle.pos, axis=vscale*vehicle.axis, color=clr.yellow)
