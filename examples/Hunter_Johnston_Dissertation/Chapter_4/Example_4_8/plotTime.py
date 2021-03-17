from util import Lagrange, getL1L2

import numpy as np
import tqdm
import pickle

from tfc.utils import MakePlot

## TEST PARAMETERS: ***************************************************
sol = pickle.load(open('data/timingLyap_CP_L1.pickle','rb'))

MS = 10
width = 0.0085      # the width of the bars: can also be len(x) sequence


p1 = MakePlot('Jacobi Constant','Computation Time [s]')

p1.ax[0].bar(sol['C'],sol['tLoss'],width, label='Loss function')
p1.ax[0].bar(sol['C'],sol['tJac'],width,bottom=sol['tLoss'],label='Jacobian')
p1.ax[0].bar(sol['C'],sol['tLS'],width,bottom=sol['tLoss']+sol['tJac'],label='Least-squares')



p1.ax[0].grid(True)
delta = 0.5*(sol['C'][0]-sol['C'][1])
p1.ax[0].set_xlim(sol['C'].min()-delta,sol['C'].max()+delta)

p1.ax[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.14),
          ncol=4, fancybox=True, shadow=True)
p1.fig.subplots_adjust(left=0.11, bottom=0.11, right=0.89, top=0.85)
p1.PartScreen(12.,7.)
p1.show()
# p1.save('figures/timeBreakdown')
