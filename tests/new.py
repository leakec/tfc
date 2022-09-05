import jax.numpy as np
from tfc import utfc
from tfc.utils import egrad, LS

# Domain circle of radius 1 around 1+1.j
# Problem: f'(x) = f(x) s.t. f(1+1.j) = 5.0 + 0.j
# Real solution: 5.0 * exp(z) / exp(1+1.j)

real = lambda x: 5.0 * np.exp(x) / np.exp(1+1.j)

a = np.sin(np.pi/4.)
x0 = 1-a + (1-a)*1.j
xf = 1+a + (1+a)*1.j
tfc = utfc(100, 1, 60, basis="CP", x0=x0, xf=xf, backend="Python")
H = tfc.H
x = tfc.x


g = lambda x,xi: np.dot(H(x),xi)
u = lambda x,xi: g(x,xi) + 5.0 - g((1+1.j)*np.ones_like(x),xi)
ud = egrad(u)
L = lambda xi,x: ud(x,xi) - u(x,xi)

xi = np.zeros(H(x).shape[1])

xi = LS(xi,L,x,constant_arg_nums=[1])

r = np.linspace(0.,1.,10).reshape((1,10))
th = np.linspace(0.,2*np.pi,10).reshape((10,1))
x = r*np.sin(th)
y = r*np.cos(th)*1.j
test = (x+y).flatten() + 1. + 1.j
