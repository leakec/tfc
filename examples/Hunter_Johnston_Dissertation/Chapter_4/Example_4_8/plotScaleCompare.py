from util import Lagrange, getL1L2

import numpy as np
import tqdm
import pickle

from matplotlib.animation import FuncAnimation

from tfc.utils import MakePlot

Lpt = 'L2'

if Lpt == 'L1':
    file1 ='Lyap_CP_L1'
    file2 ='nsLyap_CP_L1'
else:
    file1 ='Lyap_CP_L2'
    file2 ='nsLyap_CP_L2'

## TEST PARAMETERS: ***************************************************
tfc = pickle.load(open('data/' + file1 + '.pickle','rb'))[Lpt]
nsc = pickle.load(open('data/' + file2 + '.pickle','rb'))


## Compute L1 & L2 Locations
m_E = 5.9724e24
m_M = 7.346e22
mu = m_M/(m_M + m_E)
L1, L2 = getL1L2(mu)
r1 = lambda x: np.sqrt( (x+mu)**2 )
r2 = lambda x: np.sqrt( (x+mu-1.)**2 )  # m2 to (x,y,z)
Jc = lambda x: (x**2) + 2.*(1.-mu)/r1(x) + 2.*mu/r2(x) + (1.-mu)*mu

MS = 10
p1 = MakePlot('Jacobi Constant','Computation Time [s]')
p1.ax[0].plot(tfc['C'],tfc['time'],'ro', label='scaled', markersize = MS)
p1.ax[0].plot(nsc['C'],nsc['time'],'ko', label='non-scaled', markersize = MS)
p1.ax[0].axvline(x=Jc(L1), color='b', label='E(L1)', linestyle = '--')
p1.ax[0].axvline(x=Jc(L2), color='g', label='E(L2)', linestyle = '--')
p1.ax[0].grid(True)

p1.ax[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.14),
          ncol=4, fancybox=True, shadow=True)
p1.fig.subplots_adjust(left=0.11, bottom=0.11, right=0.89, top=0.85)
p1.PartScreen(9.,6.)
p1.show()
# p1.save('figures/compTime' + file1)

# p2 = MakePlot('Jacobi Constant','Iterations')
# p2.ax[0].plot(tfc['C'],tfc['iter'],'rx', label='scaled', markersize = MS)
# p2.ax[0].axvline(x=Jc(L1), color='b', label='E(L1)', linestyle = '--')
# p2.ax[0].axvline(x=Jc(L2), color='g', label='E(L2)', linestyle = '--')
# p2.ax[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.14),
#           ncol=3, fancybox=True, shadow=True)
# p2.fig.subplots_adjust(left=0.11, bottom=0.11, right=0.89, top=0.85)
# p2.ax[0].grid(True)
# p2.PartScreen(9.,6.)
# p2.show()
# p2.save('figures/iterations'+ file1)

tfcRes = np.zeros(len(tfc['C']))
nscRes = np.zeros(len(nsc['C']))
for i in range(len(tfcRes)):
    tfcRes[i] = np.max(tfc['res'][:,i])
    nscRes[i] = np.max(nsc['res'][:,i])

p3 = MakePlot('Jacobi Constant','Residuals')
p3.ax[0].plot(tfc['C'],tfcRes,'ro', label='scaled', markersize = MS)
p3.ax[0].plot(nsc['C'],nscRes,'ko', label='non-scaled', markersize = MS)
p3.ax[0].axvline(x=Jc(L1), color='b', label='E(L1)', linestyle = '--')
p3.ax[0].axvline(x=Jc(L2), color='g', label='E(L2)', linestyle = '--')
# p3.ax[0].set_ylim(1e-15,5e-13)
p3.ax[0].set_yscale('log')
p3.ax[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.14),
          ncol=4, fancybox=True, shadow=True)
p3.fig.subplots_adjust(left=0.11, bottom=0.11, right=0.89, top=0.85)
p3.ax[0].grid(True)
p3.PartScreen(9.,6.)
p3.show()
# p3.save('figures/NSresiduals'+ file1)


# p4 = MakePlot('x','y',zlabs='z')
# p4.ax[0].plot3D(L1,0.,0.,'ko', markersize=4)
# p4.ax[0].plot3D(L2,0.,0.,'ko', markersize=4)
# p4.ax[0].plot3D(1.-mu,0.,0.,'ko', markersize=10)
# # p4.ax[0].plot3D(-mu,0.,0.,'bo', markersize=20)
#
# for i in range(len(tfc['C'])):
#     line1 = p4.ax[0].plot3D(tfc['sol'][:,0,i],tfc['sol'][:,1,i],0*tfc['sol'][:,1,i],'b')
# p4.ax[0].grid(True)
# dela = 0.5
# p4.ax[0].set_xlabel(r'$x$',labelpad=10)
# p4.ax[0].set_ylabel(r'$y$',labelpad=10)
# p4.ax[0].set_zlabel(r'$z$',labelpad=10)
# p4.ax[0].set_xlim(1.-dela, 1.+dela)
# p4.ax[0].set_ylim(-dela, dela)
# p4.ax[0].set_zlim(-dela, dela)
#
# p4.PartScreen(9.,6.)
# p4.show()
# p4.save('figures/traj' + file1)

# for ii in range(0, 60, 10):
    # p4.ax[0].view_init(20, ii)
    # p4.save('figures/gif/traj' + str(ii), fileType = 'png')
