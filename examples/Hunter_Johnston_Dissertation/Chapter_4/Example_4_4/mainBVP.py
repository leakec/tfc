# Main run file for the BVP solution
# Hunter Johnston - Texas A&M University
# Updated: 15 Mar 2021
##################################################################
from tfc.utils import MakePlot

import numpy as onp
import tqdm


from BVP_tfc import BVP_tfc
from BVP_spectral import BVP_spectral
##################################################################

## TEST PARAMETERS ******************************************************************
N = 100
iterMax = 10
tol = 1e-15
mMax = 25

## TEST START ********************************************************************

m = onp.arange(2,mMax + 1,1)

# TFC - CP
errCP = onp.ones_like(m) * onp.nan
resCP = onp.ones_like(m) * onp.nan
for i in tqdm.trange(len(m)):
    err, res, _ = BVP_tfc(N, m[i], 'CP', iterMax, tol)
    errCP[i] = err
    resCP[i] = res


# TFC - ELM - Sigmoid
errES = onp.ones_like(m) * onp.nan
resES = onp.ones_like(m) * onp.nan
for i in tqdm.trange(len(m)):
    err, res, _ = BVP_tfc(N, m[i], 'ELMSigmoid', iterMax, tol)
    errES[i] = err
    resES[i] = res

# Spectral - CP
errSC = onp.ones_like(m) * onp.nan
resSC = onp.ones_like(m) * onp.nan
for i in tqdm.trange(len(m)):
    err, res, _ = BVP_spectral(N, m[i], 'CP', iterMax, tol)
    errSC[i] = err
    resSC[i] = res

# ELM - Sigmoid
errSS = onp.ones_like(m) * onp.nan
resSS = onp.ones_like(m) * onp.nan
for i in tqdm.trange(len(m)):
    err, res, _ = BVP_spectral(N, m[i], 'ELMSigmoid', iterMax, tol)
    errSS[i] = err
    resSS[i] = res


# Plot
MS = 12

p1 = MakePlot(r'Number of basis functions ($m$)',r'$L_2|y_{approx} - y_{true}|$')
p1.ax[0].plot(m,errCP,'r*', markersize=MS, label='TFC - CP')
p1.ax[0].plot(m,errSC,'k*', markersize=MS, label='Spectral - CP')
p1.ax[0].plot(m,errES,'rx', markersize=MS, label='TFC - Sigmoid')
p1.ax[0].plot(m,errSS,'kx', markersize=MS, label='Spectral - Sigmoid')
p1.ax[0].grid(True)
p1.ax[0].set_yscale('log')
p1.ax[0].set_ylim([1e-16, 1.])
# p1.ax[0].legend()
p1.ax[0].legend(loc='best', fontsize='small')
p1.PartScreen(7.,6.)
p1.show()
# p1.save('figures/BVP_TFC')


## TFC vs Spec comparison on accuracy
p2 = MakePlot(r'Number of basis functions ($m$)',r'Accuracy Gain')
p2.ax[0].plot(m,onp.log10(errSC/errCP),'k*', markersize=MS,label='CP')
p2.ax[0].grid(True)
# p4.ax[0].legend(loc='best', fontsize='small')
p2.PartScreen(7.,6.)
p2.show()
# p2.save('figures/BVP_compare')
