import sympy as sp
from sympy.core.function import AppliedUndef
from .types import ConstraintOperators, Exprs, Union, Any
from .TFCUtils import TFCPrint


class CeSolver:
    """
    Constrained expression solver.

    This class solves constrained expressions for you.

    Parameters
    ----------
    C : ConstraintOperators
        This is a tuple or list constraint operators. Each element in the
        iterable should be a Python function that takes in a sympy function,
        such as `g(x)`, and outputs that function evaluated in the same way
        as the function in the constraint. For example, if the constraint was
        u(3) = 0, then the assocaited constraint operator would be:
        `lambda u: = u.subs(x,3)`.
    kappa : Exprs
        This is a tuple or list of the kappa portion of each constraint. For the
        example u(3) = 0, the kappa portion is simply 0. Note, the standard
        Python 0 should not be used, but rather, sympy.re(0).
    s : Exprs
        This is a tuple or list of the support functions. These should be given
        in terms of sympy symbols and constants. For example, if we wanted to
        use the constant function x = 1 as a support function, then we would
        use sympy.re(1) in this iterable.
    g : Union[AppliedUndef, Any]
        This is the free function used in the constrained expression. For example,
        `g(x)`.

    References
    ----------
    The algorithm used here is given at 26:13 of this video: https://www.youtube.com/watch?v=uisOZVBHA2U&t=1573s

    Examples
    --------
    Consider the constraints `u(0) = 2` and `u_x(2) = 1` where `u_x` is the derivative of `u(x)` with respect to `x`.
    Moreover, suppose we want to use `g(x)` as the free function.

    import sympy as sp
    from tfc.utils import CeSolver

    x = sp.Symbol("x")
    y = sp.Symbol("y")
    u = sp.Function("u")
    g = sp.Function("g")

    C = [lambda u: u.subs(x,0), lambda u: sp.diff(u,x).subs(x,2)]
    K = [sp.re(2), sp.re(1)]
    s = [sp.re(1), x]

    cs = CeSolver(C,K,s, g(x))
    ce = cs.ce

    In the above code example, `ce` is the constrained expression that satisfies these constraints.
    """

    def __init__(self, C: ConstraintOperators, kappa: Exprs, s: Exprs, g: Union[AppliedUndef, Any]):
        self._C = C
        self._K = kappa
        self._s = s
        self._g = g
        self._ce_stale = True

    @property
    def ce(self) -> Any:
        """
        Constrained expression.

        Returns
        -------
        Any:
            Constrained expression.
        """
        if self._ce_stale:
            self._solveCe()
            self._ce_stale = False
        return self._ce

    @ce.setter
    def ce(self, ce: Any):
        """
        Sets the constrained expression to the user-supplied value.

        Parameters
        ----------
        ce: Any
            Sympy representation of the constrained expression.
        """
        self._ce_stale = False
        self._ce = ce

    def _solveCe(self) -> None:
        """
        Solves the constrained expression and stores it in self.ce
        """
        rho = sp.Matrix([kappa - self._C[k](self._g) for k, kappa in enumerate(self._K)])
        S = sp.Matrix([[c(s) for c in self._C] for s in self._s])
        alpha = S.inv()
        s_vec = sp.Matrix([s for s in self._s])
        phi = s_vec.transpose() * alpha
        self._ce = self._g + (phi * rho)[0]

    def checkCe(self) -> bool:
        """
        Checks the constrained expression stored in the class against the stored constraints.

        Return
        ------
        bool:
            Returns True if the constraint expression satisfies the constraints and false otherwise.
        """
        checks = [c(self.ce) == k for c, k in zip(self._C, self._K)]
        ret = True
        for k, check in enumerate(checks):
            if not check:
                TFCPrint.Error(
                    f"Expected result to be {self._K[k]}, but got {self._C[k](self.ce)}."
                )
                ret = False
        return ret
