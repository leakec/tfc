from util import Lagrange, getL1L2

import numpy as np
import tqdm
import pickle

from matplotlib.animation import FuncAnimation

from tfc.utils import MakePlot

file1 ='Halo_CP_L1_N'
file2 ='Halo_CP_L2_N'
## TEST PARAMETERS: ***************************************************
sol1 = pickle.load(open('data/' + file1 + '.pickle','rb'))
sol2 = pickle.load(open('data/' + file2 + '.pickle','rb'))
sol = {'L1':sol1['L1'], 'L2':sol2['L2'], }

m_E = 5.9724e24
m_M = 7.346e22
mu = m_M/(m_M + m_E)


Res1 = np.max(np.abs(sol['L1']['res']), axis=0)
Res2 = np.max(np.abs(sol['L2']['res']), axis=0)

a = np.where(Res1 < 1e-4)
b = np.where(Res2 < 1e-4)


a = a[0][0:37]
b = b[0][0:32]

# a = a[0]
# b = b[0]

## Compute L1 & L2 Locations
L1, L2 = getL1L2(mu)


r1 = lambda x: np.sqrt( (x+mu)**2 )
r2 = lambda x: np.sqrt( (x+mu-1.)**2 )  # m2 to (x,y,z)
Jc = lambda x: (x**2) + 2.*(1.-mu)/r1(x) + 2.*mu/r2(x) + (1.-mu)*mu

MS = 10
p1 = MakePlot('Jacobi Constant','Computation Time [s]')
p1.ax[0].plot(sol['L1']['C'][a],sol['L1']['time'][a],'bo', label='L1', markersize = MS)
p1.ax[0].plot(sol['L2']['C'][b],sol['L2']['time'][b],'rX', label='L2', markersize = MS)
p1.ax[0].axvline(x=Jc(L1), color='b', label='E(L1)', linestyle = '--')
p1.ax[0].axvline(x=Jc(L2), color='r', label='E(L2)', linestyle = '--')

p1.ax[0].legend()
p1.ax[0].grid(True)
p1.PartScreen(9.,6.)
p1.show()
# p1.save('figures/compTime' + file1)

p2 = MakePlot('Jacobi Constant','Iterations')
p2.ax[0].plot(sol['L1']['C'][a],sol['L1']['iter'][a],'bo', label='L1', markersize = MS)
p2.ax[0].plot(sol['L2']['C'][b],sol['L2']['iter'][b],'rX', label='L2', markersize = MS)
p2.ax[0].axvline(x=Jc(L1), color='b', label='E(L1)', linestyle = '--')
p2.ax[0].axvline(x=Jc(L2), color='r', label='E(L2)', linestyle = '--')
p2.ax[0].legend()
p2.ax[0].grid(True)
p2.PartScreen(9.,6.)
p2.show()
# p2.save('figures/iterations'+ file1)


p3 = MakePlot('Jacobi Constant','Residuals')
p3.ax[0].plot(sol['L1']['C'][a],Res1[a],'bo', label='L1', markersize = MS)
p3.ax[0].plot(sol['L2']['C'][b],Res2[b],'rX', label='L2', markersize = MS)

# p3.ax[0].axhline(y=np.max([1.05e-12,1.16e-13]), color='k', label='Diff. Cor. (max error)', linestyle = '--')
# p3.ax[0].axhline(y=np.min([6.29e-14,4.999e-15]), color='k', label= 'Diff. Cor. (min error)', linestyle = '--')

p3.ax[0].axvline(x=Jc(L1), color='b', label='E(L1)', linestyle = '--')
p3.ax[0].axvline(x=Jc(L2), color='r', label='E(L2)', linestyle = '--')
# p3.ax[0].set_ylim(1e-15,5e-13)
p3.ax[0].set_yscale('log')
p3.ax[0].legend()
p3.ax[0].grid(True)
p3.PartScreen(9.,6.)
p3.show()
# p3.save('figures/residuals'+ file1)



p4 = MakePlot('x','y',zlabs='z')
p4.ax[0].plot3D(L1,0.,0.,'ko', markersize=4)
p4.ax[0].plot3D(L2,0.,0.,'ko', markersize=4)
p4.ax[0].plot3D(1.-mu,0.,0.,'ko', markersize=10)
# p4.ax[0].plot3D(-mu,0.,0.,'bo', markersize=20)

for i in range(0,len(a)):
    line1 = p4.ax[0].plot3D(sol['L1']['sol'][:,0,a[i]],sol['L1']['sol'][:,1,a[i]],sol['L1']['sol'][:,2,a[i]],'b',alpha=0.45)
for i in range(0,len(b)):
    line2 = p4.ax[0].plot3D(sol['L2']['sol'][:,0,b[i]],sol['L2']['sol'][:,1,b[i]],sol['L2']['sol'][:,2,b[i]],'b',alpha=0.45)
p4.ax[0].grid(True)
dela = 0.25
p4.ax[0].set_xlabel(r'$x$',labelpad=10)
p4.ax[0].set_ylabel(r'$y$',labelpad=10)
p4.ax[0].set_zlabel(r'$z$',labelpad=10)
p4.ax[0].set_xlim(1.-dela, 1.+dela)
p4.ax[0].set_ylim(-dela, dela)
p4.ax[0].set_zlim(-dela, dela)

p4.PartScreen(9.,6.)
p4.show()
# p4.save('figures/traj' + file1)

p5 = MakePlot('x','z')
p5.ax[0].plot(L1,0.,'ko', markersize=4)
p5.ax[0].plot(L2,0.,'ko', markersize=4)
p5.ax[0].plot(1.-mu,0.,'ko', markersize=10)

for i in range(0,len(a)):
    line1 = p5.ax[0].plot(sol['L1']['sol'][:,0,a[i]],sol['L1']['sol'][:,2,a[i]],'b',alpha=0.45)
for i in range(0,len(b)):
    line2 = p5.ax[0].plot(sol['L2']['sol'][:,0,b[i]],sol['L2']['sol'][:,2,b[i]],'b',alpha=0.45)
p5.ax[0].grid(True)
p5.ax[0].set_xlabel(r'$x$',labelpad=10)
p5.ax[0].set_ylabel(r'$z$',labelpad=10)
p5.ax[0].set_xlim(0.8, 1.2)
p5.ax[0].set_ylim(-0.21, 0.21)

p5.PartScreen(9.,6.)
p5.show()
# p5.save('figures/trajXZ' + file1)
