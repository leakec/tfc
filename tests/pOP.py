import numpy as np

def CP(x,deg,d=0):
    N = np.size(x)
    One = np.ones((N,1))
    Zero = np.zeros((N,1))
    if deg == 0:
        if d > 0:
            F = Zero
        else:
            F = One
        return F
    elif deg == 1:
        if d > 1:
            F = np.hstack((Zero,Zero))
        elif d > 0:
            F = np.hstack((Zero,One))
        else:
            F = np.hstack((One,x))
        return F
    else:
        F = np.hstack((One,x,np.zeros((N,deg-1))))
        for k in range(2,deg+1):
            F[:,k:k+1] = 2.*x*F[:,k-1:k]-F[:,k-2:k-1]
        def Recurse(dark,d,dCurr=0):
            if dCurr == d:
                return dark
            else:
                if dCurr == 0:
                    dark2 = np.hstack((Zero,One,np.zeros((N,deg-1))))
                else:
                    dark2 = np.zeros((N,deg+1))
                for k in range(2,deg+1):
                    dark2[:,k:k+1] = (2.+2.*dCurr)*dark[:,k-1:k]+2.*x*dark2[:,k-1:k]-dark2[:,k-2:k-1]
                dCurr += 1
                return Recurse(dark2,d,dCurr=dCurr)
        F = Recurse(F,d)
        return F

def LeP(x,deg,d=0):
    N = np.size(x)
    One = np.ones((N,1))
    Zero = np.zeros((N,1))
    if deg == 0:
        if d > 0:
            F = Zero
        else:
            F = One
        return F
    elif deg == 1:
        if d > 1:
            F = np.hstack((Zero,Zero))
        elif d > 0:
            F = np.hstack((Zero,One))
        else:
            F = np.hstack((One,x))
        return F
    else:
        F = np.hstack((One,x,np.zeros((N,deg-1))))
        for k in range(1,deg):
            F[:,k+1:k+2] = ((2.*k+1.)*x*F[:,k:k+1]-k*F[:,k-1:k])/(k+1.)
        def Recurse(dark,d,dCurr=0):
            if dCurr == d:
                return dark
            else:
                if dCurr == 0:
                    dark2 = np.hstack((Zero,One,np.zeros((N,deg-1))))
                else:
                    dark2 = np.zeros((N,deg+1))
                for k in range(1,deg):
                    dark2[:,k+1:k+2] = ((2.*k+1.)*((dCurr+1.)*dark[:,k:k+1]+x*dark2[:,k:k+1])-k*dark2[:,k-1:k])/(k+1.)
                dCurr += 1
                return Recurse(dark2,d,dCurr=dCurr)
        F = Recurse(F,d)
        return F

def LaP(x,deg,d=0):
    N = np.size(x)
    One = np.ones((N,1))
    Zero = np.zeros((N,1))
    if deg == 0:
        if d > 0:
            F = Zero
        else:
            F = One
        return F
    elif deg == 1:
        if d > 1:
            F = np.hstack((Zero,Zero))
        elif d > 0:
            F = np.hstack((Zero,-One))
        else:
            F = np.hstack((One,1.-x))
        return F
    else:
        F = np.hstack((One,1.-x,np.zeros((N,deg-1))))
        for k in range(1,deg):
            F[:,k+1:k+2] = ((2.*k+1.-x)*F[:,k:k+1]-k*F[:,k-1:k])/(k+1.)
        def Recurse(dark,d,dCurr=0):
            if dCurr == d:
                return dark
            else:
                if dCurr == 0:
                    dark2 = np.hstack((Zero,-One,np.zeros((N,deg-1))))
                else:
                    dark2 = np.zeros((N,deg+1))
                for k in range(1,deg):
                    dark2[:,k+1:k+2] = ((2.*k+1.-x)*dark2[:,k:k+1]-(dCurr+1.)*dark[:,k:k+1]-k*dark2[:,k-1:k])/(k+1.)
                dCurr += 1
                return Recurse(dark2,d,dCurr=dCurr)
        F = Recurse(F,d)
        return F

def HoPpro(x,deg,d=0):
    N = np.size(x)
    One = np.ones((N,1))
    Zero = np.zeros((N,1))
    if deg == 0:
        if d > 0:
            F = Zero
        else:
            F = One
        return F
    elif deg == 1:
        if d > 1:
            F = np.hstack((Zero,Zero))
        elif d > 0:
            F = np.hstack((Zero,One))
        else:
            F = np.hstack((One,x))
        return F
    else:
        F = np.hstack((One,x,np.zeros((N,deg-1))))
        for k in range(1,deg):
            F[:,k+1:k+2] = x*F[:,k:k+1]-k*F[:,k-1:k]
        def Recurse(dark,d,dCurr=0):
            if dCurr == d:
                return dark
            else:
                if dCurr == 0:
                    dark2 = np.hstack((Zero,One,np.zeros((N,deg-1))))
                else:
                    dark2 = np.zeros((N,deg+1))
                for k in range(1,deg):
                    dark2[:,k+1:k+2] = (dCurr+1.)*dark[:,k:k+1]+x*dark2[:,k:k+1]-k*dark2[:,k-1:k]
                dCurr += 1
                return Recurse(dark2,d,dCurr=dCurr)
        F = Recurse(F,d)
        return F

def HoPphy(x,deg,d=0):
    N = np.size(x)
    One = np.ones((N,1))
    Zero = np.zeros((N,1))
    if deg == 0:
        if d > 0:
            F = Zero
        else:
            F = One
        return F
    elif deg == 1:
        if d > 1:
            F = np.hstack((Zero,Zero))
        elif d > 0:
            F = np.hstack((Zero,2.*One))
        else:
            F = np.hstack((One,2.*x))
        return F
    else:
        F = np.hstack((One,2.*x,np.zeros((N,deg-1))))
        for k in range(1,deg):
            F[:,k+1:k+2] = 2.*x*F[:,k:k+1]-2.*k*F[:,k-1:k]
        def Recurse(dark,d,dCurr=0):
            if dCurr == d:
                return dark
            else:
                if dCurr == 0:
                    dark2 = np.hstack((Zero,2.*One,np.zeros((N,deg-1))))
                else:
                    dark2 = np.zeros((N,deg+1))
                for k in range(1,deg):
                    dark2[:,k+1:k+2] = 2.*(dCurr+1.)*dark[:,k:k+1]+2.*x*dark2[:,k:k+1]-2.*k*dark2[:,k-1:k]
                dCurr += 1
                return Recurse(dark2,d,dCurr=dCurr)
        F = Recurse(F,d)
        return F

def FS(x,deg,d=0):
    N = np.size(x)
    F = np.zeros((N,deg+1))
    if d == 0:
        F[:,0] = 1.
        for k in range(1,deg+1):
            g = np.ceil(k/2.)
            if k%2 == 0:
                F[:,k:k+1] = np.cos(g*x)
            else:
                F[:,k:k+1] = np.sin(g*x)
    else:
        F[:,0] = 0.
        if d%4 == 0:
            for k in range(1,deg+1):
                g = np.ceil(k/2.)
                if k%2 == 0:
                    F[:,k:k+1] = g**d*np.cos(g*x)
                else:
                    F[:,k:k+1] = g**d*np.sin(g*x)
        elif d%4 == 1:
            for k in range(1,deg+1):
                g = np.ceil(k/2.)
                if k%2 == 0:
                    F[:,k:k+1] = -g**d*np.sin(g*x)
                else:
                    F[:,k:k+1] = g**d*np.cos(g*x)
        elif d%4 == 2:
            for k in range(1,deg+1):
                g = np.ceil(k/2.)
                if k%2 == 0:
                    F[:,k:k+1] = -g**d*np.cos(g*x)
                else:
                    F[:,k:k+1] = -g**d*np.sin(g*x)
        else:
            for k in range(1,deg+1):
                g = np.ceil(k/2.)
                if k%2 == 0:
                    F[:,k:k+1] = g**d*np.sin(g*x)
                else:
                    F[:,k:k+1] = -g**d*np.cos(g*x)
    return F

def nCP(X,deg,d,nC):

    # Define functions for use in generating the CP sheet
    def MultT(vec):
        tout = np.ones((N,1))
        for k in range(dim):
            tout *= T[:,vec[k]:vec[k]+1,k]
        return tout

    def Recurse(nC,deg,dim,out,vec,n=0):
        if dim > 0:
            for x in range(deg+1):
                vec[dim] = x
                out,n = Recurse(nC,deg,dim-1,out,vec,n=n)
        else:
            for x in range(deg+1):
                vec[dim] = x
                if (any(vec>=nC) and np.sum(vec) <= deg):
                    out[:,n:n+1] = MultT(vec)
                    n+=1
        return out,n
    def RecurseBasis(nC,deg,dim,vec,n=0):
        if dim > 0:
            for x in range(deg+1):
                vec[dim] = x
                n = RecurseBasis(nC,deg,dim-1,vec,n=n)
        else:
            for x in range(deg+1):
                vec[dim] = x
                if (any(vec>=nC) and np.sum(vec) <= deg):
                    n+=1
        return n

    # Collect the Chebyshev polynomails into a 3D array
    N = X.shape[0]
    dim = X.shape[1]
    T = np.zeros((N,deg+1,dim))
    for k in range(dim):
        T[:,:,k] = CP(X[:,k:k+1],deg,d[k])

    # Calculate and store all possible combinations of the individual polynomails
    vec = np.zeros(dim,dtype=int)
    numBasis = RecurseBasis(nC,deg,dim-1,vec,n=0)
    out = np.zeros((N,numBasis))
    vec *= 0
    out,n = Recurse(nC,deg,dim-1,out,vec)

    return out


def nLeP(X,deg,d,nC):

    # Define functions for use in generating the LeP sheet
    def MultT(vec):
        tout = np.ones((N,1))
        for k in range(dim):
            tout *= T[:,vec[k]:vec[k]+1,k]
        return tout

    def Recurse(nC,deg,dim,out,vec,n=0):
        if dim > 0:
            for x in range(deg+1):
                vec[dim] = x
                out,n = Recurse(nC,deg,dim-1,out,vec,n=n)
        else:
            for x in range(deg+1):
                vec[dim] = x
                if (any(vec>=nC) and np.sum(vec) <= deg):
                    out[:,n:n+1] = MultT(vec)
                    n+=1
        return out,n
    def RecurseBasis(nC,deg,dim,vec,n=0):
        if dim > 0:
            for x in range(deg+1):
                vec[dim] = x
                n = RecurseBasis(nC,deg,dim-1,vec,n=n)
        else:
            for x in range(deg+1):
                vec[dim] = x
                if (any(vec>=nC) and np.sum(vec) <= deg):
                    n+=1
        return n

    # Collect the Chebyshev polynomails into a 3D array
    N = X.shape[0]
    dim = X.shape[1]
    T = np.zeros((N,deg+1,dim))
    for k in range(dim):
        T[:,:,k] = LeP(X[:,k:k+1],deg,d[k])

    # Calculate and store all possible combinations of the individual polynomails
    vec = np.zeros(dim,dtype=int)
    numBasis = RecurseBasis(nC,deg,dim-1,vec,n=0)
    out = np.zeros((N,numBasis))
    vec *= 0
    out,n = Recurse(nC,deg,dim-1,out,vec)

    return out


def nFS(X,deg,d,nC):

    # Define functions for use in generating the LeP sheet
    def MultT(vec):
        tout = np.ones((N,1))
        for k in range(dim):
            tout *= T[:,vec[k]:vec[k]+1,k]
        return tout

    def Recurse(nC,deg,dim,out,vec,n=0):
        if dim > 0:
            for x in range(deg+1):
                vec[dim] = x
                out,n = Recurse(nC,deg,dim-1,out,vec,n=n)
        else:
            for x in range(deg+1):
                vec[dim] = x
                if (any(vec>=nC) and np.sum(vec) <= deg):
                    out[:,n:n+1] = MultT(vec)
                    n+=1
        return out,n
    def RecurseBasis(nC,deg,dim,vec,n=0):
        if dim > 0:
            for x in range(deg+1):
                vec[dim] = x
                n = RecurseBasis(nC,deg,dim-1,vec,n=n)
        else:
            for x in range(deg+1):
                vec[dim] = x
                if (any(vec>=nC) and np.sum(vec) <= deg):
                    n+=1
        return n

    # Collect the Fourier Series values into a 3D array
    N = X.shape[0]
    dim = X.shape[1]
    T = np.zeros((N,deg+1,dim))
    for k in range(dim):
        T[:,:,k] = FS(X[:,k:k+1],deg,d[k])

    # Calculate and store all possible combinations of the individual polynomails
    vec = np.zeros(dim,dtype=int)
    numBasis = RecurseBasis(nC,deg,dim-1,vec,n=0)
    out = np.zeros((N,numBasis))
    vec *= 0
    out,n = Recurse(nC,deg,dim-1,out,vec)

    return out
