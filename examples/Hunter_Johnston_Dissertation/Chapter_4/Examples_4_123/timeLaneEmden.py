# Main run file for the Lane Emden solution
# Hunter Johnston - Texas A&M University
# Updated: 10 Mar 2021
##################################################################
from tfc.utils import MakePlot

import numpy as onp
import tqdm


from laneEmden_tfc import laneEmden_tfc
from laneEmden_spectral import laneEmden_spectral
from laneEmden_ode import laneEmden_ode
##################################################################

## TEST PARAMETERS ***************************************************************
N = 100
iterMax = 10
tol = onp.finfo(float).eps
type = 5

## TEST START ********************************************************************

if type == 0:
    m = onp.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15])
    xspan = [0., 10.]
elif type == 1:
    m = onp.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 25])
    xspan = [0., 10.]
else:
    m = onp.array([10, 20, 30, 40, 45, 50, 55, 60, 65, 70])
    xspan = [0., 10.]

# TFC - CP
errCP = onp.ones_like(m) * onp.nan
timeCP = onp.ones_like(m) * onp.nan
for i in tqdm.trange(len(m)):
    err, _, time = laneEmden_tfc(N, m[i]+2, type, xspan, 'CP', iterMax, tol)
    errCP[i] = err
    timeCP[i] = time

# Spectral - CP
errSC = onp.ones_like(m) * onp.nan
timeSC = onp.ones_like(m) * onp.nan
for i in tqdm.trange(len(m)):
    err, _, time = laneEmden_spectral(N, m[i], type, xspan, 'CP', iterMax, tol)
    errSC[i] = err
    timeSC[i] = time

## ODE - 45
tol = onp.array([3,4,5,6,7,8,9,10,12,13,2.220446049250313e-14])
err45 = onp.ones_like(tol) * onp.nan
time45 = onp.ones_like(tol) * onp.nan
for i in tqdm.trange(len(tol)):
    err, time = laneEmden_ode('RK45', xspan, type, 10.**(-tol[i]))
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
# p1.ax[0].set_ylim([1e-5, 1e-1])
# p1.ax[0].legend()
p1.ax[0].legend(loc='best', fontsize='small')
p1.PartScreen(7.,6.)
p1.show()
# p1.save('figures/laneEmden_type_' + str(type) + '_time')
