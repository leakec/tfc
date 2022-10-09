# Script produces the inequality constraint plot in Figure 2.2
# Hunter Johnston - Texas A&M University
# Updated: 9 Mar 2021
##################################################################
##################################################################
from tfc import utfc
from tfc.utils import MakePlot, step

import numpy as np
#####################################################################
nLines = 20
N = 1000
basis = 'CP'
x0 = -1.
xf =  1.

# Point constraints
y1 = 4.*np.random.rand() - 2.
y2 = 4.*np.random.rand() - 2.
y3 = 4.*np.random.rand() - 2.


## DEFINE UPPER/LOWER BOUNDS: ******************************************************************
mBnd = 7
bnd = utfc(N,0,mBnd,basis = basis, x0 = x0, xf = xf)
x = bnd.x

fu = lambda xi: np.dot(bnd.H(x),xi) + 5.
fl = lambda xi: np.dot(bnd.H(x),xi) - 5.

## DEFINE CONSTRAINED EXPRESSION: ******************************************************************
nC  = 3
m   = 15

tfc = utfc(N,nC,m,basis = basis, x0 = x0, xf = xf)
x = tfc.x

x1 = tfc.x[200:201]
x2 = tfc.x[500:501]
x3 = tfc.x[-200:-199]

phi1 = lambda x: (x2*x3)/((x1-x2)*(x1-x3)) - x*(x2+x3)/((x1-x2)*(x1-x3)) + x**2/((x1-x2)*(x1-x3))

phi2 = lambda x: (x1*x3)/((x1-x2)*(x3-x2)) + x*(x1+x3)/((x1-x2)*(x2-x3)) + x**2/((x2-x1)*(x2-x3))

phi3 = lambda x: (x1*x2)/((x1-x3)*(x2-x3)) + x*(x1+x2)/((x1-x3)*(x3-x2)) + x**2/((x1-x3)*(x2-x3))

yhat = lambda xi: np.dot(tfc.H(x),xi) \
            + phi1(x)*(y1 - np.dot(tfc.H(x1),xi)) \
            + phi2(x)*(y2 - np.dot(tfc.H(x2),xi)) \
            + phi3(x)*(y3 - np.dot(tfc.H(x3),xi)) \

y = lambda xi, xil, xiu: yhat(xi) \
            + (fu(xiu)-yhat(xi)) * step(yhat(xi) -fu(xiu)) \
            + (fl(xil)-yhat(xi)) * step(fl(xil)  - yhat(xi))


## DEFINE RANDOM COEFFICIENTS ******************************************************************
xi   =  np.random.randn(tfc.H(x).shape[1],nLines)
xiu  =  np.random.randn(bnd.H(x).shape[1])
xil  =  np.random.randn(bnd.H(x).shape[1])


## CREATE PLOTS ******************************************************************

p1 = MakePlot(r'$x$',r'$y(x)$')
for i in range(nLines):
    p1.ax[0].plot(x,y(xi[:,i],xil,xiu), linewidth = 2)

p1.ax[0].plot(x,fu(xiu),'k--', linewidth = 5)
p1.ax[0].plot(x,fl(xil),'k--', linewidth = 5)

p1.ax[0].plot(x1,y1,'ko', markersize = 10)
p1.ax[0].plot(x2,y2,'ko', markersize = 10)
p1.ax[0].plot(x3,y3,'ko', markersize = 10)

p1.ax[0].grid('True')
p1.ax[0].set_xlim(x0,xf)

p1.PartScreen(9.,6.)
p1.show()
# p1.save('Equality_And_Inequality_Constraints')
