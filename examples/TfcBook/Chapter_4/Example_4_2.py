import numpy as onp
import scipy as sp
from time import process_time

import jax.numpy as np
from jax import jacfwd, jacrev, jit
from jax.lax import fori_loop

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

# Iterate to find the solution (use a jax for loop)
X = X0
val = {'s':np.array([0.]), 'X0':X0, 'X':X, 'N':N}

def body(k,val):
    val['s'] += np.linalg.multi_dot([np.linalg.inv(np.linalg.multi_dot([val['N'].T,H(val['X']),val['N']])),val['N'].T,J(val['X'])])
    val['X'] = val['X0'] - np.dot(val['N'],val['s'])
    return val

test = jit(lambda val: fori_loop(0,8,body,val))
test(val) # Call once to force compile

# Solve the problem and time the solution
tic = process_time()
val = test(val)
val['X']['x'].block_until_ready()
toc = process_time()

# Display the results
X = val['X']
print("Final results:")
print("Solution time: {0}".format(toc-tic))
print("Function value (f): {0}".format(f(X)))
print("x: {0}".format(X['x']))
print("y: {0}".format(X['y']))
print("z: {0}".format(X['z']))
