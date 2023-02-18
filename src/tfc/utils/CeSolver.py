import sympy as sp
from sympy import Expr
from sympy.core.function import AppliedUndef
from sympy.printing.pycode import PythonCodePrinter
from sympy.simplify.simplify import nc_simplify
from .types import ConstraintOperators, Exprs, Union, Any, Literal, ConstraintOperator
from .TFCUtils import TFCPrint
from sympy import latex


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
        self._ce_stale: bool = True
        self._S_stale: bool = True
        self._alpha_stale: bool = True
        self._phi_stale: bool = True
        self._rho_stale: bool = True

    @property
    def print_type(self) -> Literal["tfc", "pretty", "latex", "str"]:
        return self._print_type

    @print_type.setter
    def print_type(self, print_type: Literal["tfc", "pretty", "latex", "str"]) -> None:
        from sympy import init_printing

        self._print_type = print_type
        if self._print_type == "tfc":
            tfc_printer = TfcPrinter()
            init_printing(pretty_print=True, pretty_printer=tfc_printer.doprint)
        elif self._print_type == "str":
            init_printing(pretty_print=False)
        elif self._print_type == "pretty":
            init_printing()
        elif self._print_type == "latex":
            from sympy import latex

            init_printing(pretty_print=True, pretty_printer=latex)
        else:
            TFCPrint.Error(
                f'print_type was specified as {print_type} but only "tfc", "pretty", "latex", and "str" are accepted.'
            )

    @property
    def ce(self) -> Any:
        """
        Constrained expression.

        Returns
        -------
        Any
            Constrained expression.
        """
        if self._ce_stale:
            self._solveCe()
            self._ce_stale = False
        return self._ce

    @ce.setter
    def ce(self, ce: Any) -> None:
        """
        Sets the constrained expression to the user-supplied value.

        Parameters
        ----------
        ce: Any
            Sympy representation of the constrained expression.
        """
        self._ce_stale = False
        self._ce = ce

    @property
    def phi(self) -> Any:
        """
        Switching functions.

        Returns
        -------
        Any
            Switching functions.
        """
        if self._phi_stale:
            s_vec = sp.Matrix([s for s in self._s])
            self._phi = s_vec.transpose() * self.alpha
            self._phi_stale = False
        return self._phi

    @phi.setter
    def phi(self, phi: Any):
        """
        Set the switching functions.

        Parameters
        ----------
        phi : Any
            The switching functions.
        """
        self._phi = phi
        self._phi_stale = False

    @property
    def alpha(self) -> sp.Matrix:
        """
        Alpha matrix (inverse of the support matrix)

        Returns
        sp.Matrix
            alpha matrix. The elements are on the field over which
            the constrained expression is defined.
        """
        if self._alpha_stale:
            self._alpha = self.S.inv()
            self._alpha_stale = False
        return self._alpha

    @property
    def S(self) -> sp.Matrix:
        """
        Support matrix.

        Returns
        sp.Matrix
            Support matrix. The elements are on the field over which
            the constrained expression is defined.
        """

        def _applyC(c, s) -> Any:
            """
            Apply the constraint operator to the switching function.

            Parameters
            ----------
            c : ConstraintOperator
                Constraint operator.
            s : Expr
                Switching function.

            Returns
            -------
            Any
                c(s), which is a number on the field over which the
                constrained expression is defined.
            """

            dark = c(s)
            if isinstance(dark, sp.Matrix) or isinstance(dark, sp.MatMul):
                dark = nc_simplify(dark)[0]
            return dark

        if self._S_stale:
            self._S = sp.Matrix([[_applyC(c, s) for s in self._s] for c in self._C])
            self._S_stale = False
        return self._S

    @property
    def rho(self) -> Any:
        """
        Projection functionals.

        Returns
        -------
        Any
            Projection functionals.
        """
        if self._rho_stale:
            self._rho = sp.Matrix(
                [sp.Add(kappa, -self._C[k](self._g)) for k, kappa in enumerate(self._K)]
            )
        return self._rho

    @property
    def s(self) -> Exprs:
        """
        Switching functions.

        Returns
        -------
        Exprs
            Support functions.
        """
        return self._s

    @s.setter
    def s(self, s: Exprs) -> None:
        """
        Set the support functions.

        Parameters
        ----------
        s : Exprs
            This is a tuple or list of the support functions. These should be given
            in terms of sympy symbols and constants. For example, if we wanted to
            use the constant function x = 1 as a support function, then we would
            use sympy.re(1) in this iterable.
        """
        self._s = s
        self._S_stale = True
        self._alpha_stale = True
        self._phi_stale = True
        self._ce_stale = True

    @property
    def kappa(self):
        """
        Kappa values.

        Returns
        -------
        Exprs
            Kappa values.
        """
        return self._K

    @kappa.setter
    def kappa(self, kappa: Exprs) -> None:
        """
        Set the kappa values.

        Parameters
        -------
        kappa : Exprs
            This is a tuple or list of the kappa portion of each constraint. For the
            example u(3) = 0, the kappa portion is simply 0. Note, the standard
            Python 0 should not be used, but rather, sympy.re(0).
        """
        self._K = kappa
        self._rho_stale = True
        self._ce_stale = True

    @property
    def C(self) -> ConstraintOperators:
        """
        Constraint operators.

        Returns
        -------
        ConstraintOperators
            Constraint operators.
        """
        return self._C

    @C.setter
    def C(self, C: ConstraintOperators) -> None:
        """
        Parameters
        ----------
        C : ConstraintOperators
            This is a tuple or list constraint operators. Each element in the
            iterable should be a Python function that takes in a sympy function,
            such as `g(x)`, and outputs that function evaluated in the same way
            as the function in the constraint. For example, if the constraint was
            u(3) = 0, then the assocaited constraint operator would be:
            `lambda u: = u.subs(x,3)`.
        """
        self._C = C
        self._S_stale = True
        self._alpha_stale = True
        self._phi_stale = True
        self._rho_stale = True
        self._ce_stale = True

    @property
    def g(self) -> Union[AppliedUndef, Any]:
        """
        Free function.

        Returns
        -------
        Union[AppliedUndef, Any]
            Free function.
        """
        return self._g

    @g.setter
    def g(self, g: Union[AppliedUndef, Any]) -> None:
        """
        Set the free function.

        Parameters
        ----------
        g : Union[AppliedUndef, Any]
            This is the free function used in the constrained expression. For example,
            `g(x)`.
        """
        self._g = g
        self._ce_stale = True
        self._rho_stale = True

    def _solveCe(self) -> None:
        """
        Solves the constrained expression and stores it in self.ce
        """
        self._ce = sp.Add(self.g, (self.phi * self.rho)[0])

    def checkCe(self) -> bool:
        """
        Checks the constrained expression stored in the class against the stored constraints.

        Return
        ------
        bool:
            Returns True if the constraint expression satisfies the constraints and false otherwise.
        """
        checks = [sp.simplify(c(self.ce)) == sp.simplify(k) for c, k in zip(self._C, self._K)]
        ret = True
        for k, check in enumerate(checks):
            if not check:
                TFCPrint.Warning(
                    f"Expected result of constraint {k+1} to be {self._K[k]}, but got {self._C[k](self.ce)}."
                )
                ret = False
        return ret


class TfcPrinter(PythonCodePrinter):
    def __init__(self, settings=None):
        # Switch math to numpy
        for k, v in self._kf.items():
            if "math" in v:
                self._kf[k] = v.replace("math", "np")
        for k, v in self._kc.items():
            if "math" in v:
                self._kc[k] = v.replace("math", "np")

        super().__init__(settings=settings)

    def _hprint_Pow(self, expr: Expr, rational: bool = False, sqrt: str = "np.sqrt"):
        """
        Override _hprint_Pow to use np.sqrt rather than math.sqrt

        Parameters
        ----------
        expr : Expr
            Expression to print.
        rational : bool
            Whether the expression is rational.
        sqrt : str
            String to print for the sqrt.

        Returns
        -------
        str
            String to print.
        """

        return super()._hprint_Pow(expr, rational=rational, sqrt=sqrt)

    def _print_Symbol(self, expr: Expr) -> str:
        """
        Add in Symbol printing function.

        Parameters
        ----------
        expr : Expr
            Symbol to print.

        Returns
        -------
        str
            String to print.
        """
        return self._print(str(expr))

    def _print_Function(self, expr: Expr) -> str:
        """
        Add in Function printing function.

        Parameters
        ----------
        expr : Expr
            Function to print.

        Returns
        -------
        str
            String to print.
        """
        return self._print(str(expr))

    def _print_Subs(self, subs: Expr) -> str:
        """
        Substitute values.

        Parameters
        ----------
        subs : Expr
            Substitution(s).

        Returns
        -------
        str
            String to print.
        """
        expr, old, new = subs.args
        expr = self._print(expr)
        for k in range(len(old)):
            expr = expr.replace(self._print(old[k]), self._print(new[k]))
        return expr

    def _print_Derivative(self, expr: Expr) -> str:
        """
        Add in derivative printing function that uses egrad.

        Parameters
        ----------
        expr : Expr
            Expression to print.

        Returns
        -------
        str
            String to print.
        """
        # Function will be the full function, e.g., g(x,y)
        # vars will be the derivative symbol and order.
        # For example, dg/dx will have vars [(x,1)]
        function, *vars = expr.args

        # Find position for each derivative
        function_vars = function.args
        position_vars = []
        for var in vars:
            ind = function_vars.index(var[0])
            position_vars.append((ind, var[1]))

        # If you want the printer to work correctly for nested
        # expressions then use self._print() instead of str() or latex().
        # See the example of nested modulo below in the custom printing
        # method section.
        name = function.func.__name__
        parenthesis_counter = 0
        ret = ""
        for pv in position_vars:
            for _ in range(pv[1]):
                ret += self._print("egrad(" + name + "," + str(pv[0]))
                parenthesis_counter += 1
        for _ in range(parenthesis_counter):
            ret += self._print(")")
        ret += "(" + "".join(self._print(i[0]) for i in vars) + ")"

        return ret
