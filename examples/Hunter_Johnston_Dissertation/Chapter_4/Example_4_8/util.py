import os,sys
sourcePath = os.path.join("..","src","build","bin")
sys.path.append(sourcePath)

import numpy as np
from scipy.optimize import fsolve

def gammaL(mu, Lpt):
    ## calculates ratio of libration point distance from closest primary to distance
    #  between two primaries [example gammaL1 = (E-L1)/AU]
    ################################################################################
    mu2 = 1 - mu

    poly1 = np.array([1., -1.*(3.-mu), (3.-2*mu),  -mu,   2.*mu,  -mu])
    poly2 = np.array([1.,     (3.-mu), (3.-2*mu),  -mu,  -2.*mu,  -mu])
    poly3 = np.array([1.,     (2.+mu), (1.+2*mu), -mu2, -2.*mu2, -mu2])

    rt1 = np.roots(poly1)
    rt2 = np.roots(poly2)
    rt3 = np.roots(poly3)

    GAMMAS = np.zeros(3)
    for i in range(0,5):
        if np.isreal(rt1[i]):
            GAMMAS[0] = np.real(rt1[i])
        if np.isreal(rt2[i]):
            GAMMAS[1] = np.real(rt2[i])
        if np.isreal(rt3[i]):
            GAMMAS[2] = np.real(rt3[i])
    return GAMMAS[Lpt-1]

def rich(mu, Az, Lpt, ns='N', npts=1):

    # Uses Richardson's 3rd order model for constructing an orbit about the
    # points L1, L2, L3
    #
    # Inputs:
    #   mu      :   mass paramter
    #   Az      :   Z-amplitude of Halo orbit in GAMMA units.
    #   Lpt     :   1, 2, or 3 --> specified Lagrange point
    #   NS      :   'N' -->(1) northern halo; 'S' --> (3) southern halo
    #   npts    :   number of output points for halo orbit
    #
    #
    # Outputs:
    #   tt      :   time
    #   rvi     :   r,v state for time tt(i)
    #   period  :   period of orbit in radians
    #
    # Written based on rich.m MATLAB code updated 11/20/2018: Martin Lo

    # Gives initial conditions (r0,v0) for a halo orbit about Lpt=(1,2, or 3) with
    # amplitude Az (in units of the Lpt-nearest primary distance).
    # NS= 1 is northern(z>0, Class I), NS= 3 is southern(z<0, Class II)
    # Returns the 3x1 position matrix r0 and the 3x1 velocity matrix v0
    # r0 and v0 are transformed into the CR3BP with the LARGER mass on the left
    # npts = number of points on halo to return, equally spaced by time.

    #  CONVENTION
    #
    #                 L4
    #
    #    L3-----M1-------L1---M2---L2         M1=1-mu, M2=mu
    #
    #                 L5
    ################################################################################
    if ns == 'N':
        NS = 1.
    elif ns =='S':
        NS = 3.
    else:
        NS = 1.
        print('Improper input for NS, defaulting to Northern NS = 1')

    gamma = gammaL(mu, Lpt)
    if Lpt == 1:
        won = 1.
        primary = 1.-mu
    elif Lpt == 2:
        won = -1.
        primary = 1.-mu
    else:
        won = 1.
        primary = -mu

    c = np.zeros(4)
    if Lpt == 3:
        for N in range(2,5):
            c[N-1] = (1./gamma**3)*( 1.-mu + (-primary*gamma**(N+1))/((1.+gamma)**(N+1)) )
    else:
        for N in range(2,5):
            c[N-1] = (1./gamma**3)*((won**N)*mu+((-1.)**N)*((primary)*gamma**(N+1))/((1.+(-won)*gamma)**(N+1)) )

    polylambda = np.array([1., 0., (c[1]-2.), 0., -(c[1]-1.)*(1.+2.*c[1]) ])
    lambdaroots = np.roots(polylambda); # lambda = frequency of

    ## put np.real on lambdaroots to supress complex warning
    if Lpt==3:
        lam = np.real(lambdaroots[0])
    else:
        lam = np.real(lambdaroots[0])

    k   = 2.*lam/(lam**2 + 1. - c[1])

    dela = lam**2 - c[1]

    d1  = ((3.*lam**2)/k)*(k*( 6.*lam**2 -1.) - 2.*lam)
    d2  = ((8.*lam**2)/k)*(k*(11.*lam**2 -1.) - 2.*lam)

    a21 = (3.*c[2]*(k**2 - 2.))/(4.*(1. + 2.*c[1]))
    a22 = 3.*c[2]/(4.*(1. + 2.*c[1]))
    a23 = -(3.*c[2]*lam/(4.*k*d1))*( 3.*(k**3)*lam - 6.*k*(k-lam) + 4.)
    a24 = -(3.*c[2]*lam/(4.*k*d1))*( 2. + 3.*k*lam )

    b21 = -(3.*c[2]*lam/(2.*d1))*(3.*k*lam - 4.)
    b22 = 3.*c[2]*lam/d1
    d21 = -c[2]/(2.*lam**2)

    a31 = -(9.*lam/(4.*d2))*(4.*c[2]*(k*a23 - b21) + k*c[3]*(4. + k**2)) + ((9.*lam**2 + 1. -c[1])/(2.*d2))*(3.*c[2]*(2.*a23 - k*b21) + c[3]*(2. + 3.*k**2))
    a32 = -(1./d2)*( (9.*lam/4.)*(4.*c[2]*(k*a24 - b22) + k*c[3]) + 1.5*(9.*lam**2 + 1. - c[1])*( c[2]*(k*b22 + d21 - 2*a24) - c[3]) )

    b31 = (.375/d2)*( 8.*lam*(3.*c[2]*(k*b21 - 2.*a23) - c[3]*(2. + 3.*k**2)) + (9.*lam**2. + 1. + 2.*c[1])*(4.*c[2]*(k*a23 - b21) + k*c[3]*(4. + k**2)) )
    b32 = (1./d2)*( 9.*lam*(c[2]*(k*b22 + d21 - 2.*a24) - c[3]) + .375*(9.*lam**2 + 1. + 2.*c[1])*(4.*c[2]*(k*a24 - b22) + k*c[3]) )

    d31 = (3./(64.*lam**2))*(4.*c[2]*a24 + c[3])
    d32 = (3./(64.*lam**2))*(4.*c[2]*(a23- d21) + c[3]*(4. + k**2))

    s1  = (1./(2.*lam*(lam*(1.+k**2) - 2.*k)))*( 1.5*c[2]*(2.*a21*(k**2 - 2.)-a23*(k**2 + 2.) - 2.*k*b21) - .375*c[3]*(3.*k**4 - 8.*k**2 + 8.) )
    s2  = (1./(2.*lam*(lam*(1.+k**2) - 2.*k)))*( 1.5*c[2]*(2.*a22*(k**2 - 2.)+a24*(k**2 + 2.) + 2.*k*b22 + 5.*d21) + .375*c[3]*(12. - k**2) )

    a1  = -1.5*c[2]*(2.*a21+ a23 + 5.*d21) - .375*c[3]*(12.-k**2)
    a2  =  1.5*c[2]*(a24-2.*a22) + 1.125*c[3]

    l1 = a1 + 2.*(lam**2)*s1
    l2 = a2 + 2.*(lam**2)*s2

    # ADDITIONAL TERMS FROM GEOMETRY CENTER PAPER
    b33 = -k/(16.*lam)*(12.*c[2]*(b21-2.*k*a21+k*a23)+3.*c[3]*k*(3.*k**2-4.)+16.*s1*lam*(lam*k-1.))
    b34 = -k/(8.*lam)*(-12.*c[2]*k*a22+3.*c[3]*k+8.*s2*lam*(lam*k-1))
    b35 = -k/(16.*lam)*(12.*c[2]*(b22+k*a24)+3.*c[3]*k)

    deltan  = 2 - NS
    Ax      = np.sqrt( (-dela - l2*Az**2)/l1 )
    omg     = 1+s1*Ax**2+s2*Az**2
    freq    =lam*omg
    period  =np.abs(2*np.pi/freq)


    rvi = np.zeros((npts,6))
    ss = np.zeros((npts,1))
    if npts > 1:
        dtau1= 2.*np.pi/(npts-1.)
    else:
        dtau1= 2.*np.pi

    tau1 = 0.

    for i in range(0,npts):
        x = a21*Ax**2 + a22*Az**2 - Ax*np.cos(tau1) + (a23*Ax**2 - a24*Az**2)*np.cos(2.*tau1) + (a31*Ax**3 - a32*Ax*Az**2)*np.cos(3.*tau1)

        y = k*Ax*np.sin(tau1) + (b21*Ax**2 - b22*Az**2)*np.sin(2.*tau1) + (b31*Ax**3 - b32*Ax*Az**2)*np.sin(3.*tau1)

        y_plus = (b33*Ax**3 + b34*Ax*Az**2 - b35*Ax*Az**2)*np.sin(tau1);
        y = y + y_plus;     # ADD EXTRA TERMS FROM G.C. PAPER

        z = deltan*Az*np.cos(tau1) + deltan*d21*Ax*Az*(np.cos(2.*tau1) - 3.) + deltan*(d32*Az*Ax**2 - d31*Az**3)*np.cos(3.*tau1)

        xdot = freq*Ax*np.sin(tau1) - 2.*freq*(a23*Ax**2-a24*Az**2)*np.sin(2.*tau1) - 3.*freq*(a31*Ax**3 - a32*Ax*Az**2)*np.sin(3.*tau1)

        ydot = freq*(k*Ax*np.cos(tau1) + 2.*(b21*Ax**2 - b22*Az**2)*np.cos(2.*tau1) + 3.*(b31*Ax**3 - b32*Ax*Az**2)*np.cos(3.*tau1))

        ydot_plus = freq*(b33*Ax**3 + b34*Ax*Az**2 - b35*Ax*Az**2)*np.cos(tau1)
        ydot = ydot_plus + ydot # ADD EXTRA TERMS FROM G.C. PAPER

        zdot = - freq*deltan*Az*np.sin(tau1) - 2.*freq*deltan*d21*Ax*Az*np.sin(2.*tau1) - 3.*freq*deltan*(d32*Az*Ax**2 - d31*Az**3)*np.sin(3.*tau1);

        rvi[i] = gamma*np.array([(primary+gamma*(-won+x))/gamma, y, z, xdot, ydot, zdot])
        ss[i]   = tau1/freq
        tau1 = tau1 + dtau1

    return ss, rvi, period


def rich3_lyap(mu, Ax, Lpt, npts=1):

    # Uses Richardson's 3rd order model for constructing an orbit about the
    # points L1, L2, L3
    #
    #
    # Gives initial conditions (r0,v0) for a lyapunov orbit about Lpt=(1,2, or 3)
    # with amplitude Ax (in units of the Lpt-nearest primary distance).
    #
    # Returns the 3x1 position matrix r0 and the 3x1 velocity matrix v0
    # r0 and v0 are transformed into the CR3BP with the LARGER mass on the left
    # npts = number of points on halo to return, equally spaced by time.

    #  CONVENTION
    #
    #                 L4
    #
    #    L3-----M1-------L1---M2---L2         M1=1-mu, M2=mu
    #
    #                 L5
    ################################################################################
    gamma = gammaL(mu, Lpt)
    if Lpt == 1:
        won = 1.
        primary = 1.-mu
    elif Lpt == 2:
        won = -1.
        primary = 1.-mu
    else:
        won = 1.
        primary = -mu

    c = np.zeros(4)
    if Lpt == 3:
        for N in range(2,5):
            c[N-1] = (1./gamma**3)*( 1.-mu + (-primary*gamma**(N+1))/((1.+gamma)**(N+1)) )
    else:
        for N in range(2,5):
            c[N-1] = (1./gamma**3)*((won**N)*mu+((-1.)**N)*((primary)*gamma**(N+1))/((1.+(-won)*gamma)**(N+1)) )

    polylambda = np.array([1., 0., (c[1]-2.), 0., -(c[1]-1.)*(1.+2.*c[1]) ])
    lambdaroots = np.roots(polylambda); # lambda = frequency of

    ## put np.real on lambdaroots to supress complex warning
    if Lpt==3:
        lam = np.real(lambdaroots[0])
    else:
        lam = np.real(lambdaroots[0])

    k   = 2.*lam/(lam**2 + 1. - c[1])

    # dela = lam**2 - c[1]

    d1  = ((3.*lam**2)/k)*(k*( 6.*lam**2 -1.) - 2.*lam)
    d2  = ((8.*lam**2)/k)*(k*(11.*lam**2 -1.) - 2.*lam)

    a21 = (3.*c[2]*(k**2 - 2.))/(4.*(1. + 2.*c[1]))
    a22 = 3.*c[2]/(4.*(1. + 2.*c[1]))
    a23 = -(3.*c[2]*lam/(4.*k*d1))*( 3.*(k**3)*lam - 6.*k*(k-lam) + 4.)
    a24 = -(3.*c[2]*lam/(4.*k*d1))*( 2. + 3.*k*lam )

    b21 = -(3.*c[2]*lam/(2.*d1))*(3.*k*lam - 4.)
    b22 = 3.*c[2]*lam/d1
    d21 = -c[2]/(2.*lam**2)

    a31 = -(9.*lam/(4.*d2))*(4.*c[2]*(k*a23 - b21) + k*c[3]*(4. + k**2)) + ((9.*lam**2 + 1. -c[1])/(2.*d2))*(3.*c[2]*(2.*a23 - k*b21) + c[3]*(2. + 3.*k**2))
    a32 = -(1./d2)*( (9.*lam/4.)*(4.*c[2]*(k*a24 - b22) + k*c[3]) + 1.5*(9.*lam**2 + 1. - c[1])*( c[2]*(k*b22 + d21 - 2*a24) - c[3]) )

    b31 = (.375/d2)*( 8.*lam*(3.*c[2]*(k*b21 - 2.*a23) - c[3]*(2. + 3.*k**2)) + (9.*lam**2. + 1. + 2.*c[1])*(4.*c[2]*(k*a23 - b21) + k*c[3]*(4. + k**2)) )
    b32 = (1./d2)*( 9.*lam*(c[2]*(k*b22 + d21 - 2.*a24) - c[3]) + .375*(9.*lam**2 + 1. + 2.*c[1])*(4.*c[2]*(k*a24 - b22) + k*c[3]) )

    d31 = (3./(64.*lam**2))*(4.*c[2]*a24 + c[3])
    d32 = (3./(64.*lam**2))*(4.*c[2]*(a23- d21) + c[3]*(4. + k**2))

    s1  = (1./(2.*lam*(lam*(1.+k**2) - 2.*k)))*( 1.5*c[2]*(2.*a21*(k**2 - 2.)-a23*(k**2 + 2.) - 2.*k*b21) - .375*c[3]*(3.*k**4 - 8.*k**2 + 8.) )
    s2  = (1./(2.*lam*(lam*(1.+k**2) - 2.*k)))*( 1.5*c[2]*(2.*a22*(k**2 - 2.)+a24*(k**2 + 2.) + 2.*k*b22 + 5.*d21) + .375*c[3]*(12. - k**2) )

    a1  = -1.5*c[2]*(2.*a21+ a23 + 5.*d21) - .375*c[3]*(12.-k**2)
    a2  =  1.5*c[2]*(a24-2.*a22) + 1.125*c[3]

    # l1 = a1 + 2.*(lam**2)*s1
    # l2 = a2 + 2.*(lam**2)*s2

    # ADDITIONAL TERMS FROM GEOMETRY CENTER PAPER
    b33 = -k/(16.*lam)*(12.*c[2]*(b21-2.*k*a21+k*a23)+3.*c[3]*k*(3.*k**2-4.)+16.*s1*lam*(lam*k-1.))
    b34 = -k/(8.*lam)*(-12.*c[2]*k*a22+3.*c[3]*k+8.*s2*lam*(lam*k-1))
    b35 = -k/(16.*lam)*(12.*c[2]*(b22+k*a24)+3.*c[3]*k)

    # deltan = 1; # MODIFIED FROM NS!!
    Az = 0     # MODIF

    omg = 1.+s1*Ax**2+s2*Az**2
    freq=lam*omg
    period=np.abs(2.*np.pi/freq) # MODIFIED

    rvi = np.zeros((npts,6))
    ss = np.zeros((npts,1))
    if npts > 1:
        dtau1= 2.*np.pi/(npts-1.)
    else:
        dtau1= 2.*np.pi

    tau1 = 0.

    for i in range(0,npts):
        x = a21*Ax**2 + a22*Az**2 - Ax*np.cos(tau1) + (a23*Ax**2 - a24*Az**2)*np.cos(2.*tau1) + (a31*Ax**3 - a32*Ax*Az**2)*np.cos(3.*tau1)

        y = k*Ax*np.sin(tau1) + (b21*Ax**2 - b22*Az**2)*np.sin(2.*tau1) + (b31*Ax**3 - b32*Ax*Az**2)*np.sin(3.*tau1)
        z = 0.

        y_plus = (b33*Ax**3 + b34*Ax*Az**2 - b35*Ax*Az**2)*np.sin(tau1);
        y = y + y_plus;     # ADD EXTRA TERMS FROM G.C. PAPER

        xdot = freq*Ax*np.sin(tau1) - 2.*freq*(a23*Ax**2-a24*Az**2)*np.sin(2.*tau1) - 3.*freq*(a31*Ax**3 - a32*Ax*Az**2)*np.sin(3.*tau1)
        ydot = freq*(k*Ax*np.cos(tau1) + 2.*(b21*Ax**2 - b22*Az**2)*np.cos(2.*tau1) + 3.*(b31*Ax**3 - b32*Ax*Az**2)*np.cos(3.*tau1))

        zdot = 0.

        ydot_plus = freq*(b33*Ax**3 + b34*Ax*Az**2 - b35*Ax*Az**2)*np.cos(tau1)
        ydot = ydot_plus + ydot # ADD EXTRA TERMS FROM G.C. PAPER

        rvi[i] = gamma*np.array([(primary+gamma*(-won+x))/gamma, y, z, xdot, ydot, zdot])
        ss[i]   = tau1/freq
        tau1 = tau1 + dtau1

    return ss, rvi, period, Ax

def getL1L2(mu):
    L1 = 1. - (mu/3.)**(1./3.)
    L2 = 1. + (mu/3.)**(1./3.)
    return L1, L2

def getJacobi(S,mu):
    S = S.flatten()
    x = S[0]
    y = S[1]
    z = S[2]
    xd = S[3]
    yd = S[4]
    zd = S[5]
    R1 = np.sqrt( (x + mu)**2 + y**2 + z**2 )
    R2 = np.sqrt( (x + mu - 1.)**2 + y**2 + z**2 )

    C = (x**2 + y**2) + 2.*(1.-mu)/R1 + 2.*mu/R2 + (1.-mu)*mu - (xd**2 + yd**2 + zd**2)

    return C

def Lagrange(mu):
    ## DESCRIPTION: ******************************************************************************************
    # Determines and plots the location of Lagrange Points
    # M2 is smaller primary body, M1 is larger, R is distance between
    # Returns values in in non dimensional space
    # Sun Earth arg=[1.9890e30, 5.9720e24, 149600000];
    # Earh Moon arg=[5.9720e24, 7.34767309e22, 384400];
    # Sun Jupiter arg=[1.9890e30, 1.8983e27];

    L = np.zeros((2,5))
    fun = lambda x: x*(x+mu)**2*(x-1.+mu)**2-(1.-mu)*(x-1.+mu)**2+mu*(x+mu)**2
    L[0,0] = fsolve(fun, 0.)

    fun = lambda x: -x*(x+mu)**2*(x-1.+mu)**2+(1.-mu)*(x-1.+mu)**2+mu*(x+mu)**2
    L[0,1] = fsolve(fun, 0.)

    fun = lambda x: x*(x+mu)**2*(x-1.+mu)**2+(1.-mu)*(x-1.+mu)**2+mu*(x+mu)**2
    L[0,2] = fsolve(fun, -1.)

    L[:,3] = np.array([.5-mu, np.sqrt(3)/2])
    L[:,4] = np.array([.5-mu, -np.sqrt(3)/2])

    return L
    ## *******************************************************************************************************
