import sympy as sp
from tfc.utils import CeSolver

# Common symbols used by many tests
x = sp.Symbol("x")
y = sp.Symbol("y")
u = sp.Function("u")
u1 = sp.Function("u1")
g = sp.Function("g")

def test_univariate():
    # This test checks we can solve univariate constrained expressions.
    # Constraints
    # u(0) = 2 and u_x(2) = 1

    # Solve for the cosntrained expression.
    C = [lambda u: u.subs(x, 0), lambda u: sp.diff(u, x).subs(x, 2)]
    K = [sp.re(2), sp.re(1)]
    s = [sp.re(1), x]
    cs = CeSolver(C, K, s, g(x))

    # Assert that the constraind expression satisfies the constraints
    assert(cs.checkCe())

    # Assert that the constrained expression used as the free function
    # works
    cs2 = CeSolver(C,K,s,cs.ce)
    assert(cs2.checkCe())

def test_multivariate():
    # Solve for the multivariate constraints 
    # u(0,y)=0, u(1,y)=cos(y), and u(x,0)=u(x,2π)

    # Solve for the constrained expresssion that satisifes the x constraints
    Cx = [lambda u: u.subs(x,0), lambda u: u.subs(x,1)]
    Kx = [sp.re(0), sp.cos(y)]
    sx = [sp.re(1), x]
    csx = CeSolver(Cx, Kx, sx, g(x,y))
    cex = csx.ce
    assert(csx.checkCe())

    # Solve for the constrained expresssion that satisifes the y constraints
    Cy = [lambda u: u.subs(y,0) - u.subs(y,sp.re(2*sp.pi))]
    Ky = [sp.re(0)]
    sy = [y]
    csy = CeSolver(Cy, Ky, sy, u1(x,y))
    assert(csy.checkCe())

    # Solve for the full constrained expression
    csy.g = cex
    ce_full = csy.ce

    # Set each CeSolver to use ce_full and check their constraints
    csx.ce = ce_full

    assert(csx.checkCe())
    assert(csy.checkCe())
