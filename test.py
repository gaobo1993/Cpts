import numpy as np
import pylab as pl
from math import *

 
def sigmoid(x, alpha=1):
    return [ 1.0/(1+exp(-x[i]*alpha)) for i in range(len(x)) ]

d = np.linspace(-10.0, 10.0, num=100)

alphaList = [0, -0.5, -1, -2, -3, -4, -5, -8]

pl.figure()
plot2 = [pl.plot(d, sigmoid(d, alphaList[i]), label=r'$\alpha$ is '+str(alphaList[i])) for i in range(len(alphaList))]
pl.legend(loc=1)
pl.title('Sigmoid')
pl.xlabel('d')
pl.ylabel('y')
pl.xlim(-10.0, 10.0)# set axis limits
pl.ylim(-0.2, 1.2)
pl.grid()

pl.savefig('./figure/sigmoid.png', bbox_inches='tight')
pl.show()

