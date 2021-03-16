import os,sys
sourcePath = os.path.join("..","..","..","src","build","bin")
sys.path.append(sourcePath)

import numpy as np
import tqdm

## TEST PARAMETERS: ***************************************************
Ntest = 10000
a = 1000
b = 500

data = {'R0': np.zeros((Ntest,3)), 'V0': np.zeros((Ntest,3))}
## RUN TEST: **********************************************************

for i in tqdm.trange(Ntest):
    ang = 2.* np.pi * np.random.rand()
    r = np.random.rand() * (a*b)/np.sqrt(a**2*np.sin(ang)**2 + b**2*np.cos(ang)**2)

    data['R0'][i,:] = np.array([-2000. + r*np.cos(ang),\
                            0.    + r*np.sin(ang),\
                            1500. + (100.*np.random.rand()-100.)])

    th = np.pi * np.random.rand() - np.pi/2.

    data['V0'][i,:] = np.array([100. * np.cos(th),\
                           100. * np.sin(th),\
                           -75. + (10.*np.random.rand() - 10.)])

## END: **************************************************************
# import pickle
# with open('data/EOL_IC.pickle', 'wb') as handle:
#     pickle.dump(data, handle)

# Line to import data in other file
# sol = pickle.load(open('data/EOL_IC.pickle','rb'))
