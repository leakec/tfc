# Script produces the inequality constraint plot in Figure 2.1
# Hunter Johnston - Texas A&M University
# Updated: 9 Mar 2021
##################################################################
##################################################################
from tfc import utfc
from tfc.utils import MakePlot, step

import numpy as np
##################################################################
nLines = 20
N = 1000
basis = 'CP'
x0 = -1.
xf =  1.


## DEFINE UPPER/LOWER BOUNDS: ******************************************************************
mBnd = 7
bnd = utfc(N,0,mBnd,basis = basis, x0 = x0, xf = xf)
x = bnd.x

fu = lambda xi: np.dot(bnd.H(x),xi) + 5.
fl = lambda xi: np.dot(bnd.H(x),xi) - 5.

## DEFINE CONSTRAINED EXPRESSION: ******************************************************************
nC  = 0
m   = 15

tfc = utfc(N,nC,m,basis = basis, x0 = x0, xf = xf)
x = tfc.x

yhat = lambda xi: np.dot(tfc.H(x),xi)

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

p1.ax[0].grid('True')
p1.ax[0].set_xlim(x0,xf)

p1.PartScreen(9.,6.)
p1.show()
# p1.save('Inequality_Constraints')
