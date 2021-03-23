import numpy as onp
import jax.numpy as np

from tfc import utfc
from tfc.utils import TFCDict, NLLS, egrad, MakePlot, step
from tfc.utils.Latex import table

soln = lambda x,Pe: (1.-np.exp(Pe*(x-1.)))/(1.-np.exp(-Pe))

# Constants used in the differential equation:
tol = 1e-13

xI = 0.
xf = 1.
yi = 1.
yf = 0.

xpBound = 1.-1e-3

def CalculateSolution(Pe):
    
    # Create the TFC Class:
    N = 200
    m = 190
    nC = 2
    tfc = utfc(N,nC,m,basis='LeP',x0=0.,xf=1.)
    x = tfc.x

    # Get the Chebyshev polynomials
    H = tfc.H
    H0 = H(np.array([0.]))
    Hf = H(np.array([1.]))

    # Create the constraint expression and its derivatives
    y = lambda x,xi: np.dot(H(x),xi)+(1.-x)*(1.-np.dot(H0,xi))-x*np.dot(Hf,xi)
    yd = egrad(y)
    ydd = egrad(yd)

    L = lambda xi: ydd(x,xi)-Pe*yd(x,xi)

    # Calculate the solution
    zXi = np.zeros(H(x).shape[1])
    xi,it = NLLS(zXi,L)

    # Create the test set:
    N = 1000
    xTest = np.linspace(0.,1.,N)
    err = np.abs(y(xTest,xi)-soln(xTest,Pe))
    return np.max(err), np.mean(err)

def CalculateSolutionSplit(Pe):

    if Pe > 1e3:
        xpBoundL = 0.+1.e-3
        xpBoundU = 1.-1e-3
    else:
        xpBoundL = 0.+1.e-1
        xpBoundU = 1.-1e-1

    # Create the ToC Class:
    N = 200
    m = 190
    nC = 3
    tfc = utfc(N,nC,m,basis='LeP',x0=-1.,xf=1.)

    # Get the Chebyshev polynomials
    H = tfc.H
    dH = tfc.dH
    H0 = H(np.array([-1.]))
    Hf = H(np.array([1.]))
    Hd0 = dH(np.array([-1.]))
    Hdf = dH(np.array([1.]))

    # Create the constraint expression and its derivatives
    z = tfc.z

    xp = lambda xi: xi['xpHat']+(xpBoundU-xi['xpHat'])*step(xi['xpHat']-xpBoundU)+(xpBoundL-xi['xpHat'])*step(xpBoundL-xi['xpHat'])

    c1 = lambda xi: 2./(xp(xi))
    c2 = lambda xi: 2./(1.-xp(xi))

    x1 = lambda z,xi: (z+1.)/c1(xi)
    x2 = lambda z,xi: (z+1.)/c2(xi)+xp(xi)

    y1 = lambda z,xi: np.dot(H(z),xi['xi1'])+(1.-2.*z+z**2)/4.*(1.-np.dot(H0,xi['xi1']))\
                                   +(3.+2.*z-z**2)/4.*(xi['y']-np.dot(Hf,xi['xi1']))\
                                   +(-1.+z**2)/2.*(xi['yd']/c1(xi)-np.dot(Hdf,xi['xi1']))
    ydz1 = egrad(y1,0)
    yddz1 = egrad(ydz1,0)
    yd1 = lambda z,xi: ydz1(z,xi)*c1(xi)
    ydd1 = lambda z,xi: yddz1(z,xi)*c1(xi)**2

    y2 = lambda z,xi: np.dot(H(z),xi['xi2'])+(3.-2.*z-z**2)/4.*(xi['y']-np.dot(H0,xi['xi2']))\
                                     +(1.-z**2)/2.*(xi['yd']/c2(xi)-np.dot(Hd0,xi['xi2']))\
                                     +(1.+2.*z+z**2)/4.*(0.-np.dot(Hf,xi['xi2']))
    ydz2 = egrad(y2,0)
    yddz2 = egrad(ydz2,0)
    yd2 = lambda z,xi: ydz2(z,xi)*c2(xi)
    ydd2 = lambda z,xi: yddz2(z,xi)*c2(xi)**2

    # Solve the problem
    xi = TFCDict({'xi1':onp.zeros(H(z).shape[1]),'xi2':onp.zeros(H(z).shape[1]),'xpHat':onp.array([0.99]),'y':onp.array([0.]),'yd':onp.array([0.])})

    L1 = lambda xi: ydd1(z,xi)-Pe*yd1(z,xi)
    L2 = lambda xi: ydd2(z,xi)-Pe*yd2(z,xi)
    L = lambda xi: np.hstack([L1(xi),L2(xi)])

    xi,it = NLLS(xi,L)

    # Create the test set:
    N = 1000
    z = np.linspace(-1.,1.,N)

    # Calculate the error and return the results
    X = np.hstack([x1(z,xi),x2(z,xi)])
    Y = np.hstack([y1(z,xi),y2(z,xi)])
    err = np.abs(Y-soln(X,Pe))
    return np.max(err), np.mean(err)

err = np.block([[*CalculateSolution(1.),*CalculateSolutionSplit(1.)],
                [*CalculateSolution(10.**6),*CalculateSolutionSplit(10.**6)]])
tab = table.SimpleTable(err,form='%.2e')
print(tab)

#: Analytical solution plots
x = np.linspace(0.,1.,1000)
y1 = soln(x,1.)
y2 = soln(x,10.**6)

p = MakePlot(r'$x$',r'$y$')
p.ax[0].plot(x,y1,'k',label=r'$P_e = 1$')
p.ax[0].plot(x,y2,color=(76./256.,0.,153./256.),label=r'$P_e = 10^6$')
p.ax[0].legend()
p.PartScreen(8,7)
p.show()
