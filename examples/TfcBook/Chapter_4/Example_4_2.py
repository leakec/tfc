import numpy as onp
import scipy as sp
import jax.numpy as np
from jax import jacfwd, jacrev

from tfc.utils import TFCDict, Latex

# Initial solution
X0 = TFCDict({'x':np.array([2./3.]),
              'y':np.array([1./3.]),
              'z':np.array([1./3.])})

# Create function, Jacobian, and Hessian
f = lambda X: np.squeeze(X['x']*X['y']*(X['x']*X['y']+6.*X['y']-8.*X['x']-48.)+X['z']**2-8.*X['z']+9.*X['y']**2-72.*X['y']+16.*X['x']**2+96.*X['x']+160)
J = lambda X: np.hstack([val for val in jacfwd(f)(X).values()])
H = lambda X: np.hstack([val for val in jacrev(J)(X).values()])

# Equality constraint
A = np.array([[1.,2.,-1.],[1.,0.,1.]])
b = np.array([[1.],[1.]])
N = sp.linalg.null_space(A)

# Create table to store results
tab = onp.zeros([9,6])
tab[0,:] = np.hstack([0,X0.toArray(),f(X0),np.linalg.norm(np.dot(A,X0.toArray())-b)])

# Iterate to find the solution
s = 0
X = X0
for k in range(8):
    s += np.linalg.multi_dot([np.linalg.inv(np.linalg.multi_dot([N.T,H(X),N])),N.T,J(X)])
    X = X0 - np.dot(N,s)
    tab[k+1,:] = np.hstack([k+1,X.toArray(),f(X),np.linalg.norm(np.dot(A,X.toArray())-b)])

# Create table in latex
colHeader = [r'Iteration $k$',r'$x$',r'$y$',r'$z$',r'$f(\B{x}_k)$',r'L_2(A\B{x}}_k-\B{b})']
table = Latex.table.SimpleTable(tab,form='%.4e',colHeader=colHeader)

# Display the results
print("Results, raw numpy:")
print(tab)

print("\nResults, latex:")
print(table)
