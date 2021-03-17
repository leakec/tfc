from util import rich3_lyap, getJacobi, Lagrange
import numpy as np

from scipy import integrate

from tfc.utils import MakePlot

from time import process_time as timer
import tqdm

def PCR3BP(t,x):
    m_E = 5.9724e24
    m_M = 7.346e22
    mu = m_M/(m_M + m_E)

    r1 = np.sqrt( (x[0]+mu)**2    + x[1]**2 )
    r2 = np.sqrt( (x[0]-1.+mu)**2 + x[1]**2 )
    Omega_x = x[0]-(1.-mu)*(x[0]+mu)/r1**3-mu*(x[0]-1.+mu)/r2**3
    Omega_y = x[1]-(1.-mu)*x[1]/r1**3-mu*x[1]/r2**3

    # STM
    Uxx = 1.-(1.-mu)*(1./r1**3-3.*(x[0]+mu)**2/r1**5)-(mu)*(1./r2**3-3.*(x[0]-1.+mu)**2/r2**5);
    Uxy = (1.-mu)*3*x[1]*(x[0]+mu)/r1**5 + mu*3*x[1]*(x[0]-1.+mu)/r2**5;
    Uyy = 1.-(1.-mu)*(1./r1**3-3.*x[1]**2/r1**5)-(mu)*(1./r2**3-3.*x[1]**2/r2**5);


    A11 = np.zeros((2,2))
    A12 = np.eye(2)
    A21 = np.array([[Uxx, Uxy],[Uxy, Uyy]])
    A22 = np.array([[0., 2.],[-2., 0.]])

    F_tilda = np.block([[A11, A12],[A21, A22]])

    phi = np.reshape(x[4:20],(4,4))

    dphi = F_tilda @ phi
    dphi = dphi.flatten()

    dy = np.zeros((21, 1))
    dy[0] = x[2]
    dy[1] = x[3]
    dy[2] =  2.*x[3] + Omega_x
    dy[3] = -2.*x[2] + Omega_y
    dy[4:20] = np.reshape(dphi,(16,1))


    dy = r2*dy      # d/dtau
    dy[20] = r2;    # tracks time

    return dy.flatten()
    ## *******************************************************************************************************

def target_lyap(Lpt, Cstar, mu, iterMax):
    ## DESCRIPTION: ******************************************************************************************
    # This algorithm will find a planar lyapunov orbit at the specified energy level.
    #
    # INPUT:
    #           Lpt         - Lagrange Points (1 or 2)
    #           C           - desired Jacobi constant
    #
    # OUTPUT:
    #
    #           x0          - initial state
    #           period      - period [sec]
    #           period_tau  - period [regularized]
    #           res         - energy residual
    #*******************************************************************************************************
    tol     = 2.220446049250313e-14
    tol_sym = 2.220446049250313e-14
    tol_res = 2.220446049250313e-14

    # find initial guess with richardson
    _, rvi, period, Ax = rich3_lyap(mu, 0.2, Lpt)
    x0 = np.hstack(( rvi[0,0:2], rvi[0,3:5] ))

    # find lyapunov orbit with differential corrector
    rvi = np.reshape(np.hstack((x0[0:2], 0., x0[2:4], 0.)),(1, 6))
    C_desired = getJacobi(rvi,mu)
    dC0 = (Cstar - C_desired)

    x0, period, period_tau, _ = diff_cor_lyap_p(x0, period, Ax, mu, C_desired)


    ## Go through loop
    crit_iters = 5.
    step_up = 3./2.
    step_down = 2./3.
    max_factor = 1.
    min_factor = 1./iterMax
    factor = min_factor*2.

    for i in range(0,iterMax):
        # Target energy
        dC = factor*dC0;
        if np.abs(dC) > np.abs(Cstar-C_desired):
            dC = Cstar - C_desired

        C_desired = C_desired + dC

        # Find lyapunov orbit with diferential corrector
        x0, period, period_tau, count = diff_cor_lyap_p(x0[0:4], period, Ax, mu, C_desired)

        # Adapt step
        if count > crit_iters:
            factor = factor*step_down

        if count < crit_iters:
            factor = factor*step_up

        if factor >= max_factor:
            factor = max_factor

        if factor <= min_factor:
            factor = min_factor

        rvi = np.reshape(np.hstack((x0[0:2], 0., x0[2:4], 0.)),(1, 6))
        C = getJacobi(rvi,mu)

        print(C)

        res =  abs(Cstar - C);
        if res <= tol_res:
            break

    if res > tol_res:
        print('Residual = ' + str(res))
    else:
        print('LYAP FOUND! \n')

    return x0, period, period_tau, res
    ## *******************************************************************************************************

def diff_cor_lyap_p(x0, T, Ax, mu, C_desired):
    ## DESCRIPTION: ******************************************************************************************
    # This algorithm will generate initial conditions lying on periodic orbit
    #
    # INPUT: Data from rich3, mass parameter
    #
    # OUTPUT:  Initial condition and period of Halo orbit (in seconds and tau)
    #
    def halo_e(t,y): return y[1]
    halo_e.direction = -1
    halo_e.terminal  = True

    tol     = 2.220446049250313e-14
    tol_sym = 2.220446049250313e-14
    tol_res = 1e-12

    mu1 = 1. - mu
    mu2 = mu

    phi_0 = np.ndarray.flatten(np.eye(4))
    y_init = np.hstack(( x0, phi_0, 0.))

    # Integrate
    NotPeriodic = True
    d2 = np.sqrt( (x0[0]-1.+mu)**2 + x0[1]**2 ) # time regularization
    tau_max = 2.*T/d2 # guess of maximum time

    count = 0
    while NotPeriodic:
        sol = integrate.solve_ivp(PCR3BP, [0., tau_max], y_init, method='DOP853', events = halo_e, rtol=tol, atol=tol)

        t = sol.t
        y = sol.y

        X  = y[0,-1]
        Y  = y[1,-1]
        Xd = y[2,-1]
        Yd = y[3,-1]

        # check conditons of ''bad'' solution
        if np.abs(Y) >= 1e-3:
            NotPeriodic = False
            print('Integration time exceeded!! \n')

        L = Lagrange(mu)
        d_norm2 = 1. - mu - L[0,0]
        if np.linalg.norm(y_init[0:2] - sol.y[0:2,-1]) >= 100.*Ax*d_norm2: # Stray too far
            NotPeriodic = False
            print('ESCAPED!! \n')

        # Check if symmetry conditons met
        rvi = np.array([X, Y, 0., Xd, Yd, 0.])
        C = getJacobi(rvi,mu)
        if np.abs(Xd) < tol_sym and np.abs(C-C_desired) < tol_sym:
            NotPeriodic = False
            # print('Converged! \n')
        else:
            r1 = np.sqrt( (X+mu)**2 + Y**2 )
            r2 = np.sqrt( (X-1.+mu)**2 + Y**2 )
            Omega_x = X - (1. - mu ) * (X + mu)/r1**3-mu*(X-1.+mu)/r2**3
            xdd = 2.*Yd + Omega_x;

            ## Check energy
            # Get partials (in terms of tau)
            x0  = sol.y[0,0]
            y0  = sol.y[1,0]
            vx0 = sol.y[2,0]
            vy0 = sol.y[3,0]
            den1 = ((x0-mu1)**2 + y0**2)**(1.5)
            den2 = ((x0+mu2)**2 + y0**2)**(1.5)
            C_x = 2.*x0 - 2.*mu2*(x0-mu1)/den1 - 2.*mu1*(x0+mu2)/den2;
            C_ydot = -2.*vy0;


            # Calculate M
            M_2_0_a = np.array([[y[12,-1], y[15,-1]],[C_x, C_ydot]])
            M_2_0_b = 1./Yd * np.array([[xdd],[0.]]) @ np.array([[y[8,-1], y[11,-1]]])
            M_2_0 = M_2_0_a - M_2_0_b


            ## Energy Constrained Approach
            b = -np.array([[y[2,-1]],[-C_desired + C]])

            q =  np.linalg.pinv(M_2_0) @ b
            y_init[0] = y_init[0] + q[0]
            y_init[3] = y_init[3] + q[1]

        x0 = y[:,0]
        period_tau = 2.*t[-1]
        period = 2.*y[-1,-1]

        if np.abs(period) == 0:
            print('FALSE CONVERGENCE LIKELY!!')
            period = np.abs(T);

        count += 1

    return x0, period, period_tau, count
    ## *******************************************************************************************************
## Start files
import pickle

file1 ='Lyap_CP_L1'
file2 ='Lyap_CP_L2'
## TEST PARAMETERS: ***************************************************
sol1 = pickle.load(open('data/' + file1 + '.pickle','rb'))
sol2 = pickle.load(open('data/' + file2 + '.pickle','rb'))
tfc = {'L1':sol1['L1'], 'L2':sol2['L2'], }


m_E = 5.9724e24
m_M = 7.346e22
mu = m_M/(m_M + m_E)
tol     = 2.220446049250313e-14

Lpt = 'L2'
Cstar = tfc[Lpt]['C']


if Lpt == 'L1':
    _, rvi, period, Ax = rich3_lyap(mu, 0.05, 1)
else:
    _, rvi, period, Ax = rich3_lyap(mu, 0.05, 2)

# find initial guess with richardson
x0 = np.hstack(( rvi[0,0:2], rvi[0,3:5] ))

# find lyapunov orbit with differential corrector
rvi = np.reshape(np.hstack((x0[0:2], 0., x0[2:4], 0.)),(1, 6))

dif = {'res':np.zeros(len(Cstar)),\
       'time':np.zeros(len(Cstar)),\
       'C':np.zeros(len(Cstar))}

for i in tqdm.trange(len(Cstar)):

    start = timer()
    x0, period, period_tau, _ = diff_cor_lyap_p(x0[0:4], period, Ax, mu, Cstar[i])
    dif['time'][i] = timer() - start

    sol = integrate.solve_ivp(PCR3BP, [0, period_tau], x0, method='DOP853', rtol=tol, atol=tol) #, t_eval=tspan)

    resState = np.abs(sol.y[0:4,0] - sol.y[0:4,-1])

    X  = sol.y[0,:]
    Y  = sol.y[1,:]
    Xd = sol.y[2,:]
    Yd = sol.y[3,:]

    ctemp = np.zeros(len(sol.t))
    for j in range(len(sol.t)):
        ctemp[j] = getJacobi(np.array([X[j],Y[j],0.,Xd[j],Yd[j],0.]),mu)

    resJc = np.abs(Cstar[i] - ctemp)

    dif['res'][i]   = np.max(np.hstack((resState, resJc)))
    dif['C'][i]     = Cstar[i]

## END ******************************************************************************************************
# import pickle
# with open('data/' + 'diffcor' + str(Lpt) + '.pickle', 'wb') as handle:
#     pickle.dump(dif, handle)
