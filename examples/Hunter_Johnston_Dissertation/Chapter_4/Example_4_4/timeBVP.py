# Main run file for the BVP solution (timed)
# Hunter Johnston - Texas A&M University
# Updated: 15 Mar 2021
##################################################################
from tfc.utils import MakePlot

import numpy as onp
import tqdm

from BVP_tfc import BVP_tfc
from BVP_spectral import BVP_spectral
from BVP_ode import BVP_ode
##################################################################


## TEST PARAMETERS ***************************************************************
N = 100
iterMax = 10
tol = onp.finfo(float).eps
m = onp.array([6, 8, 10, 12, 14, 16, 18, 19, 20, 21, 22])

## TEST START ********************************************************************

# TFC - CP
errCP = onp.ones_like(m) * onp.nan
timeCP = onp.ones_like(m) * onp.nan
for i in tqdm.trange(len(m)):
    err, _, time = BVP_tfc(N, m[i], 'CP', iterMax, tol)
    errCP[i] = err
    timeCP[i] = time

# Spectral - CP
errSC = onp.ones_like(m) * onp.nan
timeSC = onp.ones_like(m) * onp.nan
for i in tqdm.trange(len(m)):
    err, _, time = BVP_spectral(N, m[i], 'CP', iterMax, tol)
    errSC[i] = err
    timeSC[i] = time

## ODE - 45
tol = onp.array([3,4,5,6,7,8,9])
err45 = onp.ones_like(tol) * onp.nan
time45 = onp.ones_like(tol) * onp.nan
for i in tqdm.trange(len(tol)):
    err, time = BVP_ode('RK45', 10.**(-tol[i]))
    err45[i] = err
    time45[i] = time

# Solvers: RK45, RK23, DOP853, Radau, BDF, LSODA

# Plot
MS = 12

p1 = MakePlot(r'$L_2|y_{approx} - y_{true}|$',r'Solution Time [sec]')
p1.ax[0].plot(errCP,timeCP,'r*', markersize=MS,label='TFC - CP')
p1.ax[0].plot(errSC,timeSC,'k*', markersize=MS, label='Spectral - CP')
p1.ax[0].plot(err45,time45,'b*', markersize=MS, label='RK45')

p1.ax[0].grid(True)
p1.ax[0].set_xscale('log')
p1.ax[0].set_yscale('log')
p1.ax[0].set_ylim([1e-4, 10.])
# p1.ax[0].legend()
p1.ax[0].legend(loc='best', fontsize='small')
p1.PartScreen(7.,6.)
p1.show()
# p1.save('figures/BVP_time')
