""" Use the TFC method to solve for balloon shapes at all altitudes. """

import pickle
import pandas as pd

import numpy as onp
import jax.numpy as np
from jax import jit, jacfwd

from tfc import utfc
from tfc.utils import TFCDict, egrad, MakePlot, NllsClass

# Import the atmosheric data:
atmData = pd.read_excel("BalloonData.xlsx",header=1)

# Problem constants:
Rs = 2.75
Rz = 5.5
w = 0.095
ws = 0.215
mw_gas = 4.00e-3
mw_atm = 4.34e-2

# Calcualted constants:
As = 4.*np.pi*Rs**2
Vs = 4./3*np.pi*Rs**3

# TFC constants:
N = 100
m = 80

tol = 1.e-13
maxIter = 15

# Create the TFC Classes:
tfc = utfc(N,2,m,basis='CP',x0=-1.,xf=1.)
tfc1 = utfc(N,1,m,basis='CP',x0=-1.,xf=1.)
x = tfc.x

# Get the Chebyshev polynomials
H = tfc.H
H0 = H(np.array([-1.]))
Hf = H(np.array([1.]))

H1 = tfc1.H
H10 = H1(np.array([-1.]))

# Boundary conditions
s0 = lambda xi: Rs*xi['beta']
z0 = lambda xi: Rs*(1.-np.cos(xi['beta']))
r0 = lambda xi: Rs*np.sin(xi['beta'])
th0 = lambda xi: np.pi/2.-xi['beta']

# Use heaviside function to mimic an if condition that is JIT-able
def To1(xi,const):
    Aso = 2.*np.pi*Rs*z0(xi)
    Vso = np.pi/3.*z0(xi)**2*(3.*Rs-z0(xi))
    return const['L']+const['g']*(w+ws)*Aso+const['g']*(Vso/Vs*const['Msg']-Vso*const['atm_density'])
def To2(xi,const):
    Hq = 2.*Rs-z0(xi)
    Aso = As-2.*np.pi*Rs*Hq
    Vso = Vs-np.pi/3.*Hq**2*(3.*Rs-Hq)
    return const['L']+const['g']*(w+ws)*Aso+const['g']*(Vso/Vs*const['Msg']-Vso*const['atm_density'])
def To(xi,const):
    return np.heaviside(-(xi['beta']-np.pi),0.)*To1(xi,const)+np.heaviside(xi['beta']-np.pi,1.)*To2(xi,const)

sig0 = lambda xi,const: To(xi,const)/(2.*np.pi*r0(xi)*np.sin(xi['beta']))
q0 = lambda xi,const: 1./(sig0(xi,const)*r0(xi))

# Define scaling parameter for linear mapping
c = lambda xi: 2./(xi['ld']-s0(xi))
s = lambda x,xi: (x+1.)/c(xi)+s0(xi)

# Create the constrained expressions and their derivatives:
q = lambda x,xi,const: np.dot(H1(x),xi['xiQ'])+q0(xi,const)-np.dot(H10,xi['xiQ'])
z = lambda x,xi: np.dot(H1(x),xi['xiZ'])+z0(xi)-np.dot(H10,xi['xiZ'])
th = lambda x,xi: np.dot(H(x),xi['xiTh'])+(1.-x)/2.*(th0(xi)-np.dot(H0,xi['xiTh']))+(x+1.)/2.*(-np.pi/2.-np.dot(Hf,xi['xiTh']))
r = lambda x,xi: np.dot(H(x),xi['xiR'])+(1.-x)/2.*(r0(xi)-np.dot(H0,xi['xiR']))-(x+1.)/2.*np.dot(Hf,xi['xiR'])

dq = egrad(q)
dz = egrad(z)
dth = egrad(th)
dr = egrad(r)

# Create the residuals/Jacobians:
resTh = lambda x,xi,const: c(xi)*dth(x,xi)+q(x,xi,const)*r(x,xi)*(w*np.sin(th(x,xi))+const['b']*(z(x,xi)-z0(xi)))
resQ = lambda x,xi,const: c(xi)*dq(x,xi,const)+q(x,xi,const)**2*r(x,xi)*w*np.cos(th(x,xi))
resR = lambda x,xi: c(xi)*dr(x,xi)-np.sin(th(x,xi))
resZ = lambda x,xi: c(xi)*dz(x,xi)-np.cos(th(x,xi))
res = jit(lambda xi,const: np.hstack([resTh(x,xi,const), resQ(x,xi,const), resR(x,xi), resZ(x,xi)]))

def Jdark(x,xi,const):
    jacob = jacfwd(res,0)(xi,const)
    return np.hstack((jacob[k] for k in xi.keys()))
J = jit(lambda xi,const: Jdark(x,xi,const))

# Create the NLLS 
def cond(val):
    return np.all(np.array([
                np.max(np.abs(res(val['xi'],val['const']))) > tol,
                val['it'] < 30,
                np.max(np.abs(val['dxi'])) > tol]))
def body(val):
    val['dxi'] = -np.dot(np.linalg.pinv(J(val['xi'],val['const'])),res(val['xi'],val['const']))
    val['xi'] += val['dxi']
    val['it'] += 1
    return val

nlls = jit(lambda val: lax.while_loop(cond,body,val))

# Create plot
th2 = np.linspace(0.,2.*np.pi,num=100)
X2 = Rs*np.cos(th2)
Y2 = Rs*np.sin(th2)+Rs

p = MakePlot(r'$x\ (m)$',r'$y\ (m)$')
p.ax[0].plot(X2,Y2,'-k',label='Super Pressure Balloon')
p.ax[0].grid(True)

colors = ["r", "g", "b", "tab:gray", "m", "c", "tab:orange", "y", "tab:purple", "tab:brown", "tab:olive"]

const = {'g':0.,'L':0.,'atm_density':0.,'Msg':0.,'b':0.}
time = onp.zeros(atmData.shape[0])

for k in range(atmData.shape[0]-1,-1,-1):

    # Problem constants:
    const['g'] = atmData['Gravity (m/s^2)'][k]
    const['L'] = 208*const['g']
    const['atm_density'] = atmData['Density (kg/m^3)'][k]
    const['Msg'] = atmData['Gas Mass (kg).1'][k]

    # Calcualted constants:
    const['b'] = const['g']*const['atm_density']*(1.-mw_gas/mw_atm)

    # Initial guess
    if k == atmData.shape[0]-1:
        init = pickle.load(open('TfcInit.pkl','rb'))

        m = H(x).shape[1]
        m1 = H1(x).shape[1]

        xi = TFCDict({'xiTh':onp.zeros((m)), 'xiQ':onp.zeros((m1)), 'xiR':onp.zeros((m)),
                      'xiZ':onp.zeros((m1)), 'beta':onp.zeros((1)), 'ld':onp.zeros((1))})

        xi['beta'] = init['beta']
        xi['ld'] = init['ld']

        xp = init['x'].flatten()

        ro = onp.expand_dims(onp.interp(x,xp,init['r'].flatten()),1)
        xi['xiR'] = onp.dot(onp.linalg.pinv(jacfwd(r,1)(x,xi)['xiR']),ro.flatten()-r(x,xi))

        tho = onp.expand_dims(onp.interp(x,xp,init['th'].flatten()),1)
        xi['xiTh'] = onp.dot(onp.linalg.pinv(jacfwd(th,1)(x,xi)['xiTh']),tho.flatten()-th(x,xi))

        zo = onp.expand_dims(onp.interp(x,xp,init['z'].flatten()),1)
        xi['xiZ']= onp.dot(onp.linalg.pinv(jacfwd(z,1)(x,xi)['xiZ']),zo.flatten()-z(x,xi))

        qo = onp.expand_dims(onp.interp(x,xp,init['q'].flatten()),1)
        xi['xiQ'] = onp.dot(onp.linalg.pinv(jacfwd(q,1)(x,xi,const)['xiQ']),qo.flatten()-q(x,xi,const))
        
        # Create NLLS class
        nlls = NllsClass(xi,res,tol=tol,maxIter=maxIter,timer=True)

    # Run the NLLS
    xi,_,time[k] = nlls.run(xi,const)

    # Plots and results
    print("Altitude: "+str(atmData['Alt (km)'][k])+"km \t Norm of the residual: "+str(np.linalg.norm(res(xi,const))))

    th1 = np.linspace(0.,xi['beta'],num=100).flatten()
    x1 = -Rs*np.sin(th1)
    y1 = Rs*(1.-np.cos(th1))
    x2 = -r(x,xi)
    y2 = z(x,xi)
    X = np.hstack([x1,x2])
    Y = np.hstack([y1,y2])

    p.ax[0].plot(X,Y,linestyle='-',color=colors[k],label='Altitude: '+str(atmData['Alt (km)'][k])+' km')
    p.ax[0].plot(-X,Y,linestyle='-',color=colors[k])

# Print results and show plot
print("Average time: "+str(np.mean(time)))
p.ax[0].legend(ncol=2)
p.ax[0].axis('equal')
p.FullScreen()
p.show()
