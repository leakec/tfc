
import numpy as np
import tqdm
import pickle

from tfc.utils import MakePlot
from matplotlib.ticker import PercentFormatter



## TEST PARAMETERS: ***************************************************
tfc = pickle.load(open('data/EOL_TFC.pickle','rb'))
spe = pickle.load(open('data/EOL_Spec.pickle','rb'))

## Plot: **************************************************************
MS = 12

# import pdb; pdb.set_trace()

## Plot 1: Accuracy
bin1 = np.logspace(-17, -11, num=50, endpoint=True, base=10.0, dtype=None, axis=0)

p1 = MakePlot('Accuracy ($|L_2|$)',r'Frequency')


p1.ax[0].hist(tfc['loss'][np.where(tfc['loss'] < 1.)], bin1, edgecolor='black', linewidth=1.2, label = 'TFC',alpha = 0.75, \
weights=np.ones(len(tfc['loss'][np.where(tfc['loss'] < 1.)])) / len(tfc['loss'][np.where(tfc['loss'] < 1.)]))
p1.ax[0].yaxis.set_major_formatter(PercentFormatter(1, decimals=0, symbol='%'))

p1.ax[0].hist(spe['loss'][np.where(spe['loss'] < 1.)], bin1, edgecolor='black', linewidth=1.2, label = 'Spectral Method', alpha = 0.75, \
weights=np.ones(len(spe['loss'][np.where(spe['loss'] < 1.)])) / len(spe['loss'][np.where(spe['loss'] < 1.)]))
p1.ax[0].yaxis.set_major_formatter(PercentFormatter(1, decimals=0, symbol='%'))

p1.ax[0].set_xscale('log')
p1.ax[0].set_xlim(5e-17, 5e-15)
p1.fig.subplots_adjust(wspace=0.35, hspace=0.25)
p1.ax[0].legend()
p1.PartScreen(9.,6.)
p1.show()
# p1.save('figures/EOL_hist_L2_outerLoop')


## Plot 2: Computation time
bin2 = np.linspace(0,300,30)

p2 = MakePlot('Computation time [ms]',r'Frequency')

p2.ax[0].hist(tfc['time'][np.where(tfc['time'] < 1.)]*1000, bin2, edgecolor='black', linewidth=1.2,
label = 'TFC', alpha = 0.75, \
weights=np.ones(len(tfc['time'][np.where(tfc['time'] < 1.)])) / len(tfc['time'][np.where(tfc['time'] < 1.)]))
p2.ax[0].yaxis.set_major_formatter(PercentFormatter(1, decimals=0, symbol='%'))

p2.ax[0].hist(spe['time'][np.where(spe['time'] < 1.)]*1000, bin2, edgecolor='black', linewidth=1.2, label = 'Spectral Method', alpha = 0.75, \
weights=np.ones(len(spe['time'][np.where(spe['time'] < 1.)])) / len(spe['time'][np.where(spe['time'] < 1.)]))
p2.ax[0].yaxis.set_major_formatter(PercentFormatter(1, decimals=0, symbol='%'))

p2.ax[0].set_xlim(100, 300)
p2.fig.subplots_adjust(wspace=0.35, hspace=0.25)
p2.ax[0].legend()
p2.PartScreen(9.,6.)
p2.show()
# p2.save('figures/EOL_hist_time_outerLoop')


## Plot 3: Number of itations
bin3 = np.arange(0,50,1)

p3 = MakePlot(r'Iterations',r'Frequency')

p3.ax[0].hist(tfc['it'][np.where(tfc['time'] < 1.)], bin3, edgecolor='black', linewidth=1.2,
label = 'TFC',alpha = 0.75, \
weights=np.ones(len(tfc['it'][np.where(tfc['time'] < 1.)])) / len(tfc['it'][np.where(tfc['time'] < 1.)]))
p3.ax[0].yaxis.set_major_formatter(PercentFormatter(1, decimals=0, symbol='%'))

p3.ax[0].hist(spe['it'][np.where(spe['time'] < 1.)], bin3, edgecolor='black', linewidth=1.2, label = 'Spectral Method', alpha = 0.75, \
weights=np.ones(len(spe['it'][np.where(spe['time'] < 1.)])) / len(spe['it'][np.where(spe['time'] < 1.)]))
p3.ax[0].yaxis.set_major_formatter(PercentFormatter(1, decimals=0, symbol='%'))

p3.ax[0].set_xlim(15, 30)
p3.fig.subplots_adjust(wspace=0.35, hspace=0.25)
p3.ax[0].legend()
p3.PartScreen(9.,6.)
p3.show()
# p3.save('figures/EOL_hist_it_outerLoop')
