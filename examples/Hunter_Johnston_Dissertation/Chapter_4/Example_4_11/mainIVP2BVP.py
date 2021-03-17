
##################################################################################
import numpy as np
from tfc.utils import MakePlot

import tqdm


from IVP2BVP import IVP2BVP
##################################################################################

## TEST PARAMETERS ***************************************************************
N = 100
m = 60

iterMax = 10
tol = 1e-13
step = 15

## TEST START ********************************************************************

Y   = np.zeros((N,step))
RES = np.zeros((N,step))

gamma = np.linspace(0,1,step)


for i in tqdm.trange(len(gamma)):
    y, res, x = IVP2BVP(N, m, gamma[i], 'CP', iterMax, tol)
    Y[:,i]   = y
    RES[:,i] = res


sRes = 3.*np.std(RES,axis=1)
mRes = np.mean(RES,axis=1)


lw = 1
p1 = MakePlot(r'$x$',r'$y(x)$',zlabs = r'$\gamma$')
for i in range(0,len(gamma)):
    p1.ax[0].plot3D(x,Y[:,i],gamma[i]*np.ones_like(x),'k', linewidth=lw)
p1.ax[0].grid(True)
p1.ax[0].view_init(20, -130)
p1.ax[0].set_facecolor('white')
p1.ax[0].xaxis.labelpad = 15
p1.ax[0].yaxis.labelpad = 15
p1.ax[0].zaxis.labelpad = 15
p1.PartScreen(7.,6.)
p1.show()
# p1.save('figures/IVP2BVP_soln')

p2 = MakePlot(r'$x$',r'$|\mathbb{L}(\xi)|$')
p2.ax[0].plot(x,mRes,'k*', markersize=10*lw,label='Mean Residual')
p2.ax[0].plot(x,sRes,'r*', markersize=10*lw, label=r'Mean Residual + 3$\sigma$')

p2.ax[0].grid(True)
p2.ax[0].set_yscale('log')
p2.ax[0].legend()
p2.PartScreen(7.,6.)

p2.show()
# p2.save('figures/IVP2BVP_res')
