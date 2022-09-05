import jax.numpy as np
from tfc import mtfc
from tfc.utils import egrad, LS

# Domain square of side length 2 centered around 1+1.j
# Problem: f'(x) = f(x) s.t. f(1+1.j) = 5.0 + 0.j
# Real solution: 5.0 * exp(z) / exp(1+1.j)

real = lambda x: 5.0 * np.exp(x) / np.exp(1+1.j)

a = np.sin(np.pi/4.)
x0 = 1-a + (1-a)*1.j
xf = 1+a + (1+a)*1.j
tfc = mtfc([19,19], [1, 1], 20, basis="CP", x0=[0.+0.j, 0+0.j], xf=[2.0+0.j, 0.+2.j], backend="Python")
H = tfc.H
x = tfc.x
x = x[0] + x[1]

g = lambda x,xi: np.dot(H(np.real(x),np.imag(x)),xi)
u = lambda x,xi: g(x,xi) + 5.0 - g((1+1.j)*np.ones_like(x),xi)
ud = egrad(u)
L = lambda xi,x: ud(x,xi) - u(x,xi)

xi = np.zeros(H(np.real(x),np.imag(x)).shape[1])

xi = LS(xi,L,x,constant_arg_nums=[1])

#r = np.linspace(0.,1.,10).reshape((1,10))
#th = np.linspace(0.,2*np.pi,10).reshape((10,1))
#x = r*np.sin(th)
#y = r*np.cos(th)*1.j
#test = (x+y).flatten() + 1. + 1.j
