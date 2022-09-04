import numpy as np
import jax.numpy as jnp
from abc import ABC, abstractmethod
from numpy import typing as npt
from tfc.utils.types import uint, Number


class BasisFunc(ABC):
    """
    Python implementation of the basis function classes. These are an alternative
    to the C++ versions. They can not be JIT-ed, but they do support alternative
    types, e.g., single float, complex, etc. Even though they cannnot be JIT-ed,
    they can often be used in JIT functions, if their arguments can be removed
    from said functions. For example, when solving an ODE, oftentimes the basis
    functions can be treated as compile time constants. This can be done using
    `pejit`: see `pejit` for more details.
    """

    def __init__(
        self,
        x0: Number,
        xf: Number,
        nC: npt.NDArray,
        m: uint,
        z0: Number = 0,
        zf: Number = float("inf"),
    ) -> None:
        """
        Initialize the basis class.

        Parameters:
        -----------
        x0: Number
            Start of the problem domain.
        xf: Number
            End of the problem domain.
        nC: npt.NDArray
            Basis functions to be removed
        m: uint
            Number of basis functions.
        z0: Number
            Start of the basis function domain.
        zf: Number
            End of the basis function domain.
        """

        self._m = m
        self._nC = nC
        self._numC = len(nC)

        self._z0 = z0
        if zf == float("inf"):
            self._c = 1.0
            self._x0 = 0.0
        else:
            self._x0 = x0
            self._c = (zf - z0) / (xf - x0)

    def H(self, x: npt.NDArray, d: uint = 0, full: bool = False) -> npt.NDArray:
        """
        Returns the basis function matrix for the x with a derivative of order d.

        Parameters:
        -----------
        x: NDArray
            Input array. Values to calculate the basis function for.
        d: uint
            Order of the derivative
        full: bool
            Whether to return the full basis function set, or remove
            the columns associated with self._nC.

        Returns:
        --------
        H: NDArray
            The basis function values.
        """

        z = (x - self._x0) * self._c + self._z0
        if len(z.shape) == 1:
            z = np.expand_dims(z, 1)
        dMult = self._c**d
        F = self._Hint(z, d) * dMult
        if not full and self._numC > 0:
            F = np.delete(F, self._nC)
        return F

    @abstractmethod
    def _Hint(self, z: npt.NDArray, d: uint) -> npt.NDArray:
        """
        Internal method used to calcualte the basis function value.

        Parameters:
        -----------
        z: NDArray
            Values to calculate the basis functions for.
        d: uint
            Derivative order.

        Returns:
        --------
        H: NDArray
            Basis function values.
        """
        pass


class CP(BasisFunc):
    """
    Chebyshev polynomial basis functions.
    """

    def __init__(
        self,
        x0: Number,
        xf: Number,
        nC: npt.NDArray,
        m: uint,
    ) -> None:
        """
        Initialize the basis class.

        Parameters:
        -----------
        x0: Number
            Start of the problem domain.
        xf: Number
            End of the problem domain.
        nC: npt.NDArray
            Basis functions to be removed
        m: uint
            Number of basis functions.
        """
        super().__init__(x0, xf, nC, m, -1.0, 1.0)

    def _Hint(self, z: npt.NDArray, d: uint) -> npt.NDArray:
        """
        Internal method used to calcualte the CP basis function values.

        Parameters:
        -----------
        z: NDArray
            Values to calculate the basis functions for.
        d: uint
            Derivative order.

        Returns:
        --------
        H: NDArray
            Basis function values.
        """
        N = np.size(z)
        One = np.ones_like(z)
        Zero = np.zeros_like(z)
        if self._m == 1:
            if d > 0:
                F = Zero
            else:
                F = One
            return F
        elif self._m == 2:
            if d > 1:
                F = np.hstack((Zero, Zero))
            elif d > 0:
                F = np.hstack((Zero, One))
            else:
                F = np.hstack((One, z))
            return F
        else:
            F = np.hstack((One, z, np.zeros((N, self._m - 2), dtype=z.dtype)))
            for k in range(2, self._m):
                F[:, k : k + 1] = 2 * z * F[:, k - 1 : k] - F[:, k - 2 : k - 1]

            def Recurse(dark, d, dCurr=0):
                if dCurr == d:
                    return dark
                else:
                    if dCurr == 0:
                        dark2 = np.hstack((Zero, One, np.zeros((N, self._m - 2), dtype=z.dtype)))
                    else:
                        dark2 = np.zeros((N, self._m), dtype=z.dtype)
                    for k in range(2, self._m):
                        dark2[:, k : k + 1] = (
                            (2 + 2 * dCurr) * dark[:, k - 1 : k]
                            + 2 * z * dark2[:, k - 1 : k]
                            - dark2[:, k - 2 : k - 1]
                        )
                    dCurr += 1
                    return Recurse(dark2, d, dCurr=dCurr)

            return Recurse(F, d)


class LeP(BasisFunc):
    """
    Legendre polynomial basis functions.
    """

    def __init__(
        self,
        x0: Number,
        xf: Number,
        nC: npt.NDArray,
        m: uint,
    ) -> None:
        """
        Initialize the basis class.

        Parameters:
        -----------
        x0: Number
            Start of the problem domain.
        xf: Number
            End of the problem domain.
        nC: npt.NDArray
            Basis functions to be removed
        m: uint
            Number of basis functions.
        """
        super().__init__(x0, xf, nC, m, -1.0, 1.0)

    def _Hint(self, z: npt.NDArray, d: uint) -> npt.NDArray:
        """
        Internal method used to calcualte the LeP basis function values.

        Parameters:
        -----------
        z: NDArray
            Values to calculate the basis functions for.
        d: uint
            Derivative order.

        Returns:
        --------
        H: NDArray
            Basis function values.
        """
        N = np.size(z)
        One = np.ones_like(z)
        Zero = np.zeros_like(z)
        if self._m == 1:
            if d > 0:
                F = Zero
            else:
                F = One
            return F
        elif self._m == 2:
            if d > 1:
                F = np.hstack((Zero, Zero))
            elif d > 0:
                F = np.hstack((Zero, One))
            else:
                F = np.hstack((One, z))
            return F
        else:
            F = np.hstack((One, z, np.zeros((N, self._m - 2), dtype=z.dtype)))
            for k in range(1, self._m - 1):
                F[:, k + 1 : k + 2] = (
                    (2.0 * k + 1.0) * z * F[:, k : k + 1] - k * F[:, k - 1 : k]
                ) / (k + 1.0)

            def Recurse(dark, d, dCurr=0):
                if dCurr == d:
                    return dark
                else:
                    if dCurr == 0:
                        dark2 = np.hstack((Zero, One, np.zeros((N, self._m - 2), dtype=z.dtype)))
                    else:
                        dark2 = np.zeros((N, self._m), dtype=z.dtype)
                    for k in range(1, self._m - 1):
                        dark2[:, k + 1 : k + 2] = (
                            (2.0 * k + 1.0)
                            * ((dCurr + 1.0) * dark[:, k : k + 1] + z * dark2[:, k : k + 1])
                            - k * dark2[:, k - 1 : k]
                        ) / (k + 1.0)
                    dCurr += 1
                    return Recurse(dark2, d, dCurr=dCurr)

            return Recurse(F, d)


class LaP(BasisFunc):
    """
    Laguerre polynomial basis functions.
    """

    def _Hint(self, z: npt.NDArray, d: uint) -> npt.NDArray:
        """
        Internal method used to calcualte the LaP basis function values.

        Parameters:
        -----------
        z: NDArray
            Values to calculate the basis functions for.
        d: uint
            Derivative order.

        Returns:
        --------
        H: NDArray
            Basis function values.
        """
        N = np.size(z)
        One = np.ones_like(z)
        Zero = np.zeros_like(z)
        if self._m == 1:
            if d > 0:
                F = Zero
            else:
                F = One
            return F
        elif self._m == 2:
            if d > 1:
                F = np.hstack((Zero, Zero))
            elif d > 0:
                F = np.hstack((Zero, -One))
            else:
                F = np.hstack((One, 1.0 - z))
            return F
        else:
            F = np.hstack((One, 1.0 - z, np.zeros((N, self._m - 2), dtype=z.dtype)))
            for k in range(1, self._m - 1):
                F[:, k + 1 : k + 2] = (
                    (2.0 * k + 1.0 - z) * F[:, k : k + 1] - k * F[:, k - 1 : k]
                ) / (k + 1.0)

            def Recurse(dark, d, dCurr=0):
                if dCurr == d:
                    return dark
                else:
                    if dCurr == 0:
                        dark2 = np.hstack((Zero, -One, np.zeros((N, self._m - 2), dtype=z.dtype)))
                    else:
                        dark2 = np.zeros((N, self._m), dtype=z.dtype)
                    for k in range(1, self._m - 1):
                        dark2[:, k + 1 : k + 2] = (
                            (2.0 * k + 1.0 - z) * dark2[:, k : k + 1]
                            - (dCurr + 1.0) * dark[:, k : k + 1]
                            - k * dark2[:, k - 1 : k]
                        ) / (k + 1.0)
                    dCurr += 1
                    return Recurse(dark2, d, dCurr=dCurr)

            return Recurse(F, d)


class HoPpro(BasisFunc):
    """
    Hermite probablist polynomial basis functions.
    """

    def _Hint(self, z: npt.NDArray, d: uint) -> npt.NDArray:
        """
        Internal method used to calcualte the HoPpro basis function values.

        Parameters:
        -----------
        z: NDArray
            Values to calculate the basis functions for.
        d: uint
            Derivative order.

        Returns:
        --------
        H: NDArray
            Basis function valuesa
        """
        N = np.size(z)
        One = np.ones_like(z)
        Zero = np.zeros_like(z)
        if self._m == 1:
            if d > 0:
                F = Zero
            else:
                F = One
            return F
        elif self._m == 2:
            if d > 1:
                F = np.hstack((Zero, Zero))
            elif d > 0:
                F = np.hstack((Zero, One))
            else:
                F = np.hstack((One, z))
            return F
        else:
            F = np.hstack((One, z, np.zeros((N, self._m - 2), dtype=z.dtype)))
            for k in range(1, self._m - 1):
                F[:, k + 1 : k + 2] = z * F[:, k : k + 1] - k * F[:, k - 1 : k]

            def Recurse(dark, d, dCurr=0):
                if dCurr == d:
                    return dark
                else:
                    if dCurr == 0:
                        dark2 = np.hstack((Zero, One, np.zeros((N, self._m - 2), dtype=z.dtype)))
                    else:
                        dark2 = np.zeros((N, self._m), dtype=z.dtype)
                    for k in range(1, self._m - 1):
                        dark2[:, k + 1 : k + 2] = (
                            (dCurr + 1.0) * dark[:, k : k + 1]
                            + z * dark2[:, k : k + 1]
                            - k * dark2[:, k - 1 : k]
                        )
                    dCurr += 1
                    return Recurse(dark2, d, dCurr=dCurr)

            return Recurse(F, d)


class HoPphy(BasisFunc):
    """
    Hermite physicist polynomial basis functions.
    """

    def _Hint(self, z: npt.NDArray, d: uint) -> npt.NDArray:
        """
        Internal method used to calcualte the HoPpro basis function values.

        Parameters:
        -----------
        z: NDArray
            Values to calculate the basis functions for.
        d: uint
            Derivative order.

        Returns:
        --------
        H: NDArray
            Basis function valuesa
        """
        N = np.size(z)
        One = np.ones_like(z)
        Zero = np.zeros_like(z)
        if self._m == 1:
            if d > 0:
                F = Zero
            else:
                F = One
            return F
        elif self._m == 2:
            if d > 1:
                F = np.hstack((Zero, Zero))
            elif d > 0:
                F = np.hstack((Zero, 2.0 * One))
            else:
                F = np.hstack((One, 2.0 * z))
            return F
        else:
            F = np.hstack((One, 2.0 * z, np.zeros((N, self._m - 2), dtype=z.dtype)))
            for k in range(1, self._m - 1):
                F[:, k + 1 : k + 2] = 2.0 * z * F[:, k : k + 1] - 2.0 * k * F[:, k - 1 : k]

            def Recurse(dark, d, dCurr=0):
                if dCurr == d:
                    return dark
                else:
                    if dCurr == 0:
                        dark2 = np.hstack(
                            (Zero, 2.0 * One, np.zeros((N, self._m - 2), dtype=z.dtype))
                        )
                    else:
                        dark2 = np.zeros((N, self._m), dtype=z.dtype)
                    for k in range(1, self._m - 1):
                        dark2[:, k + 1 : k + 2] = (
                            2.0 * (dCurr + 1.0) * dark[:, k : k + 1]
                            + 2.0 * z * dark2[:, k : k + 1]
                            - 2.0 * k * dark2[:, k - 1 : k]
                        )
                    dCurr += 1
                    return Recurse(dark2, d, dCurr=dCurr)

            return Recurse(F, d)


class FS(BasisFunc):
    """
    Chebyshev polynomial basis functions.
    """

    def __init__(
        self,
        x0: Number,
        xf: Number,
        nC: npt.NDArray,
        m: uint,
    ) -> None:
        """
        Initialize the basis class.

        Parameters:
        -----------
        x0: Number
            Start of the problem domain.
        xf: Number
            End of the problem domain.
        nC: npt.NDArray
            Basis functions to be removed
        m: uint
            Number of basis functions.
        """
        super().__init__(x0, xf, nC, m, -np.pi, np.pi)

    def _Hint(self, z: npt.NDArray, d: uint) -> npt.NDArray:
        """
        Internal method used to calcualte the CP basis function values.

        Parameters:
        -----------
        z: NDArray
            Values to calculate the basis functions for.
        d: uint
            Derivative order.

        Returns:
        --------
        H: NDArray
            Basis function values.
        """
        N = np.size(z)
        F = np.zeros((N, self._m))
        if d == 0:
            F[:, 0] = 1.0
            for k in range(1, self._m):
                g = np.ceil(k / 2.0)
                if k % 2 == 0:
                    F[:, k : k + 1] = np.cos(g * z)
                else:
                    F[:, k : k + 1] = np.sin(g * z)
        else:
            F[:, 0] = 0.0
            if d % 4 == 0:
                for k in range(1, self._m):
                    g = np.ceil(k / 2.0)
                    if k % 2 == 0:
                        F[:, k : k + 1] = g**d * np.cos(g * z)
                    else:
                        F[:, k : k + 1] = g**d * np.sin(g * z)
            elif d % 4 == 1:
                for k in range(1, self._m):
                    g = np.ceil(k / 2.0)
                    if k % 2 == 0:
                        F[:, k : k + 1] = -(g**d) * np.sin(g * z)
                    else:
                        F[:, k : k + 1] = g**d * np.cos(g * z)
            elif d % 4 == 2:
                for k in range(1, self._m):
                    g = np.ceil(k / 2.0)
                    if k % 2 == 0:
                        F[:, k : k + 1] = -(g**d) * np.cos(g * z)
                    else:
                        F[:, k : k + 1] = -(g**d) * np.sin(g * z)
            else:
                for k in range(1, self._m):
                    g = np.ceil(k / 2.0)
                    if k % 2 == 0:
                        F[:, k : k + 1] = g**d * np.sin(g * z)
                    else:
                        F[:, k : k + 1] = -(g**d) * np.cos(g * z)
        return F


class ELM(BasisFunc):
    """
    Extreme learning machine abstract basis class.
    """

    def __init__(
        self,
        x0: Number,
        xf: Number,
        nC: npt.NDArray,
        m: uint,
    ) -> None:
        """
        Initialize the basis class.

        Parameters:
        -----------
        x0: Number
            Start of the problem domain.
        xf: Number
            End of the problem domain.
        nC: npt.NDArray
            Basis functions to be removed
        m: uint
            Number of basis functions.
        """
        super().__init__(x0, xf, nC, m, 0.0, 1.0)

        self._w = np.random.uniform(low=-10.0, high=10.0, size=self._m)
        self._w = self._w.reshape((1, self._m))
        self._b = np.random.uniform(low=-10.0, high=10.0, size=self._m)
        self._b = self._b.reshape((1, self._m))

    @property
    def w(self) -> npt.NDArray:
        """
        Weights of the ELM
        """
        return self._w

    @property
    def b(self) -> npt.NDArray:
        """
        Biases of the ELM
        """
        return self._b

    @w.setter
    def w(self, val: npt.NDArray) -> None:
        """
        Weights of the ELM.
        """
        if val.size == self._m:
            self._w = val
            if self._w.shape != (1, self._m):
                self._w = self._w.reshape((1, self._m))
        else:
            raise ValueError(
                f"Input array of size {val.size} was received, but size {self._m} was expected."
            )

    @b.setter
    def b(self, val: npt.NDArray) -> None:
        """
        Biases of the ELM.
        """
        if val.size == self._m:
            self._b = val
            if self._b.shape != (1, self._m):
                self._b = self._b.reshape((1, self._m))
        else:
            raise ValueError(
                f"Input array of size {val.size} was received, but size {self._m} was expected."
            )


class ELMReLU(ELM):
    def _Hint(self, z: npt.NDArray, d: uint) -> npt.NDArray:
        """
        Internal method used to calcualte the ELMRelu basis function values.

        Parameters:
        -----------
        z: NDArray
            Values to calculate the basis functions for.
        d: uint
            Derivative order.

        Returns:
        --------
        H: NDArray
            Basis function values.
        """

        if d == 0:
            return np.maximum(0.0, self._w * z + self._b)
        elif d == 1:
            return self._w * np.where(self._w * z + self._b > 0.0, 1.0, 0.0)
        else:
            return np.zeros((self._m, z.size))


class ELMSigmoid(ELM):
    def _Hint(self, z: npt.NDArray, d: uint) -> npt.NDArray:
        """
        Internal method used to calcualte the ELMSigmoid basis function values.

        Parameters:
        -----------
        z: NDArray
            Values to calculate the basis functions for.
        d: uint
            Derivative order.

        Returns:
        --------
        H: NDArray
            Basis function values.
        """

        from tfc.utils import egrad

        f = lambda x: 1.0 / (1.0 + jnp.exp(-self._w * x - self._b))

        def Recurse(dark, d, dCurr=0):
            if dCurr == d:
                return dark
            else:
                dark2 = egrad(dark)
                dCurr += 1
                return Recurse(dark2, d, dCurr=dCurr)

        return Recurse(f, d)(z).to_py()


class ELMTanh(ELM):
    def _Hint(self, z: npt.NDArray, d: uint) -> npt.NDArray:
        """
        Internal method used to calcualte the ELMTanh basis function values.

        Parameters:
        -----------
        z: NDArray
            Values to calculate the basis functions for.
        d: uint
            Derivative order.

        Returns:
        --------
        H: NDArray
            Basis function values.
        """

        from tfc.utils import egrad

        f = lambda x: jnp.tanh(self._w * x + self._b)

        def Recurse(dark, d, dCurr=0):
            if dCurr == d:
                return dark
            else:
                dark2 = egrad(dark)
                dCurr += 1
                return Recurse(dark2, d, dCurr=dCurr)

        return Recurse(f, d)(z).to_py()


class ELMSin(ELM):
    def _Hint(self, z: npt.NDArray, d: uint) -> npt.NDArray:
        """
        Internal method used to calcualte the ELMSin basis function values.

        Parameters:
        -----------
        z: NDArray
            Values to calculate the basis functions for.
        d: uint
            Derivative order.

        Returns:
        --------
        H: NDArray
            Basis function values.
        """

        from tfc.utils import egrad

        f = lambda x: jnp.sin(self._w * x + self._b)

        def Recurse(dark, d, dCurr=0):
            if dCurr == d:
                return dark
            else:
                dark2 = egrad(dark)
                dCurr += 1
                return Recurse(dark2, d, dCurr=dCurr)

        return Recurse(f, d)(z).to_py()


class ELMSwish(ELM):
    def _Hint(self, z: npt.NDArray, d: uint) -> npt.NDArray:
        """
        Internal method used to calcualte the ELMSwish basis function values.

        Parameters:
        -----------
        z: NDArray
            Values to calculate the basis functions for.
        d: uint
            Derivative order.

        Returns:
        --------
        H: NDArray
            Basis function values.
        """

        from tfc.utils import egrad

        f = lambda x: (self._w * x + self._b) / (1.0 + jnp.exp(-self._w * x - self._b))

        def Recurse(dark, d, dCurr=0):
            if dCurr == d:
                return dark
            else:
                dark2 = egrad(dark)
                dCurr += 1
                return Recurse(dark2, d, dCurr=dCurr)

        return Recurse(f, d)(z).to_py()


class nBasisFunc(BasisFunc):
    """
    Python implementation of the n-dimensional basis function classes.
    See the Python implementation of `BasisFunc` for details.
    """

    def __init__(
        self,
        x0: npt.NDArray,
        xf: npt.NDArray,
        nC: npt.NDArray,
        m: uint,
        z0: Number = 0.0,
        zf: Number = 0.0,
    ) -> None:
        """
        Initialize the basis class.

        Parameters:
        -----------
        x0: NDArray
            Start of the problem domain.
        xf: NDArray
            End of the problem domain.
        nC: NDArray
            Basis functions to be removed
        m: uint
            Number of basis functions.
        z0: Number
            Start of the basis function domain.
        zf: Number
            End of the basis function domain.
        """

        self._m = m
        self._nC = nC
        self._dim = nC.shape[0]
        self._numC = nC.shape[1]

        self._z0 = z0
        self._zf = zf
        self._x0 = x0
        if self._x0.shape != (self._dim, 1):
            self._x0 = self._x0.reshape((self._dim, 1))
        if xf.shape != (self._dim, 1):
            xf = xf.reshape((self._dim, 1))
        self._c = (zf - z0) / (xf - self._x0)

        vec = np.zeros((self._dim, 1))
        self._numBasisFunc = self._NumBasisFunc(self._dim - 1, vec, full=False)
        self._numBasisFuncFull = self._NumBasisFunc(self._dim - 1, vec, full=True)

    def _NumBasisFunc(self, dim: int, vec: npt.NDArray, n: int = 0, full: bool = False):
        if dim > 0:
            for x in range(self._m):
                vec[dim] = x
                n = self._NumBasisFunc(dim - 1, vec, n=n, full=full)
        else:
            for x in range(self._m):
                vec[dim] = x
                if full:
                    if np.sum(vec) <= self._m - 1:
                        # If the degree of the produce of univariate basis functions is less than
                        # the degree specified, then add one to the count.
                        n += 1
                else:
                    if not np.all(np.any(vec == self._nC, axis=1)) and np.sum(vec) <= self._m - 1:
                        # If at least one of the dimensions' basis functions is not a constraint
                        # and the degree of the product of univariate basis functions is less than
                        # the degree specified, add one to the count
                        n += 1
        return n

    @property
    def c(self) -> npt.NDArray:
        """
        Return the constants that map the problem domain to the basis
        function domain.

        Returns:
        --------
        npt.NDArray
            The constants that map the problem domain to the basis function
            domain.
        """

        return self._c

    def H(self, x: npt.NDArray, d: npt.NDArray, full: bool = False) -> npt.NDArray:
        """
        Returns the basis function matrix for the x with a derivative of order d.

        Parameters:
        -----------
        x: NDArray
            Input array. Values to calculate the basis function for.
            Should be size dim x N.
        d: NDArray
            Order of the derivative
        full: bool
            Whether to return the full basis function set, or remove
            the columns associated with self._nC.

        Returns:
        --------
        H: NDArray
            The basis function values.
        """

        # Check dimensions
        N = x.shape[1]
        if x.shape[0] != self._dim:
            raise ValueError(
                f"Incorrect dimension for x. Expected {self._dim} but got {z.shape[1]}."
            )

        # Convert to basis function domain
        z = (x - self._x0) * self._c + self._z0

        # Create individual basis functions for each dimension
        T = np.zeros((N, self._m, self._dim), dtype=z.dtype)
        for k in range(self._dim):
            T[:, :, k] = self._Hint(z[k : k + 1, :].T, d[k]) * self._c[k] ** d[k]

        # Define functions for use in generating the CP sheet
        def MultT(vec: npt.NDArray) -> npt.NDArray:
            tout = np.ones((N, 1), dtype=z.dtype)
            for k in range(self._dim):
                tout *= T[:, vec[k, 0] : vec[k, 0] + 1, k]
            return tout

        def Recurse(dim: int, out: npt.NDArray, vec: npt.NDArray, n: int = 0, full: bool = False):
            if dim > 0:
                for x in range(self._m):
                    vec[dim] = x
                    out, n = Recurse(dim - 1, out, vec, n=n, full=full)
            else:
                for x in range(self._m):
                    vec[dim] = x
                    if full:
                        if np.sum(vec) <= self._m - 1:
                            # If the degree of the produce of univariate basis functions is less than
                            # the degree specified, then include this vector.
                            out[:, n : n + 1] = MultT(vec)
                            n += 1
                    else:
                        if (
                            not np.all(np.any(vec == self._nC, axis=1))
                            and np.sum(vec) <= self._m - 1
                        ):
                            # If at least one of the dimensions' basis functions is not a constraint
                            # and the degree of the product of univariate basis functions is less than
                            # the degree specified, include this vector.
                            out[:, n : n + 1] = MultT(vec)
                            n += 1
            return out, n

        # Calculate and store all possible combinations of the individual basis functions
        vec = np.zeros((self._dim, 1), dtype=int)
        if full:
            out = np.zeros((N, self._numBasisFuncFull), dtype=z.dtype)
        else:
            out = np.zeros((N, self._numBasisFunc), dtype=z.dtype)
        out, _ = Recurse(self._dim - 1, out, vec, full=full)

        return out


class nCP(nBasisFunc, CP):
    """
    n-dimensional Chebyshev polynomial basis functions.
    """

    def __init__(
        self,
        x0: npt.NDArray,
        xf: npt.NDArray,
        nC: npt.NDArray,
        m: uint,
    ) -> None:
        """
        Initialize the n-dimensional CP class.

        Parameters:
        -----------
        x0: NDArray
            Start of the problem domain.
        xf: NDArray
            End of the problem domain.
        nC: NDArray
            Basis functions to be removed
        m: uint
            Number of basis functions.
        """

        nBasisFunc.__init__(self, x0, xf, nC, m, -1.0, 1.0)


class nLeP(nBasisFunc, LeP):
    """
    n-dimensional Legendre polynomial basis functions.
    """

    def __init__(
        self,
        x0: npt.NDArray,
        xf: npt.NDArray,
        nC: npt.NDArray,
        m: uint,
    ) -> None:
        """
        Initialize the n-dimensional LeP class.

        Parameters:
        -----------
        x0: NDArray
            Start of the problem domain.
        xf: NDArray
            End of the problem domain.
        nC: NDArray
            Basis functions to be removed
        m: uint
            Number of basis functions.
        """

        nBasisFunc.__init__(self, x0, xf, nC, m, -1.0, 1.0)


class nFS(nBasisFunc, FS):
    """
    n-dimensional Fourier series basis functions.
    """

    def __init__(
        self,
        x0: npt.NDArray,
        xf: npt.NDArray,
        nC: npt.NDArray,
        m: uint,
    ) -> None:
        """
        Initialize the n-dimensional FS class.

        Parameters:
        -----------
        x0: NDArray
            Start of the problem domain.
        xf: NDArray
            End of the problem domain.
        nC: NDArray
            Basis functions to be removed
        m: uint
            Number of basis functions.
        """

        nBasisFunc.__init__(self, x0, xf, nC, m, -np.pi, np.pi)


class nELM(nBasisFunc):
    """
    n-dimensional extreme learning machine abstract basis class.
    """

    def __init__(
        self,
        x0: npt.NDArray,
        xf: npt.NDArray,
        nC: npt.NDArray,
        m: uint,
        z0: Number = 0.0,
        zf: Number = 1.0,
    ) -> None:
        """
        Initialize the basis class.

        Parameters:
        -----------
        x0: NDArray
            Start of the problem domain.
        xf: NDArray
            End of the problem domain.
        nC: NDArray
            Basis functions to be removed
        m: uint
            Number of basis functions.
        z0: Number
            Start of the basis function domain.
        zf: Number
            End of the basis function domain.
        """

        self._m = m
        self._nC = nC
        self._dim = x0.size

        if np.any(self._nC != -1):
            self._numC = nC.size
        else:
            self._numC = 0

        self._z0 = z0
        self._zf = zf
        self._x0 = x0
        if self._x0.shape != (self._dim, 1):
            self._x0 = self._x0.reshape((self._dim, 1))
        if xf.shape != (self._dim, 1):
            xf = xf.reshape((self._dim, 1))
        self._c = (zf - z0) / (xf - self._x0)

        self._numBasisFunc = self._m - self._numC
        self._numBasisFuncFull = self._m

        self._w = np.random.uniform(low=-1.0, high=1.0, size=self._dim * self._m)
        self._w = self._w.reshape((self._dim, self._m))
        self._b = np.random.uniform(low=-1.0, high=1.0, size=self._m)
        self._b = self._b.reshape((1, self._m))

    @property
    def w(self) -> npt.NDArray:
        """
        Weights of the nELM
        """
        return self._w

    @property
    def b(self) -> npt.NDArray:
        """
        Biases of the nELM
        """
        return self._b

    @w.setter
    def w(self, val: npt.NDArray) -> None:
        """
        Weights of the nELM.
        """
        if val.size == self._m * self._dim:
            self._w = val
            if self._w.shape != (self._dim, self._m):
                self._w = self._w.reshape((self._dim, self._m))
        else:
            raise ValueError(
                f"Input array of size {val.size} was received, but size {self._m*self._dim} was expected."
            )

    @b.setter
    def b(self, val: npt.NDArray) -> None:
        """
        Biases of the nELM.
        """
        if val.size == self._m:
            self._b = val
            if self._b.shape != (1, self._m):
                self._b = self._b.reshape((1, self._m))
        else:
            raise ValueError(
                f"Input array of size {val.size} was received, but size {self._m} was expected."
            )

    def H(self, x: npt.NDArray, d: npt.NDArray, full: bool = False) -> npt.NDArray:
        """
        Returns the basis function matrix for the x with a derivative of order d.

        Parameters:
        -----------
        x: NDArray
            Input array. Values to calculate the basis function for.
            Should be size dim x N.
        d: NDArray
            Order of the derivative
        full: bool
            Whether to return the full basis function set, or remove
            the columns associated with self._nC.

        Returns:
        --------
        H: NDArray
            The basis function values.
        """

        # Check dimensions
        if x.shape[0] != self._dim:
            raise ValueError(
                f"Incorrect dimension for x. Expected {self._dim} but got {z.shape[1]}."
            )

        # Convert to basis function domain
        z = ((x - self._x0) * self._c + self._z0).T

        F = self._nHint(z, d)
        if not full and self._numC > 0:
            F = np.delete(F, self._nC, axis=1)
        return F

    @abstractmethod
    def _nHint(self, z: npt.NDArray, d: npt.NDArray) -> npt.NDArray:
        """
        Internal method used to calcualte the basis function value.

        Parameters:
        -----------
        z: NDArray
            Values to calculate the basis functions for.
        d: NDArray
            Derivative order.

        Returns:
        --------
        H: NDArray
            Basis function values.
        """
        pass

    def _Hint(self, z: npt.NDArray, d: uint) -> npt.NDArray:
        """
        Dummy function, this should never be called!
        """
        raise ValueError("Error: This function should never be called.")


class nELMReLU(nELM):
    def _nHint(self, z: npt.NDArray, d: npt.NDArray) -> npt.NDArray:
        """
        Internal method used to calcualte the basis function value.

        Parameters:
        -----------
        z: NDArray
            Values to calculate the basis functions for.
        d: NDArray
            Derivative order.

        Returns:
        --------
        H: NDArray
            Basis function values.
        """
        ind = -1
        zeroFlag = False
        dorder = np.sum(d)
        if dorder > 1:
            zeroFlag = True
        elif dorder == 1:
            ind = np.where(d == 1)[0][0]

        if zeroFlag:
            # Derivative order is high enough that everything is zeros
            return np.zeros((z.shape[0], self._m))
        elif ind != -1:
            # We have a derivative on only one variable
            return (
                self._c[ind]
                * self._w[ind : ind + 1, :]
                * np.where(np.dot(z, self._w) + self._b > 0.0, 1.0, 0.0)
            )
        else:
            return np.maximum(0.0, np.dot(z, self._w) + self._b)


class nELMSin(nELM):
    def _nHint(self, z: npt.NDArray, d: npt.NDArray) -> npt.NDArray:
        """
        Internal method used to calcualte the basis function value.

        Parameters:
        -----------
        z: NDArray
            Values to calculate the basis functions for.
        d: NDArray
            Derivative order.

        Returns:
        --------
        H: NDArray
            Basis function values.
        """

        from tfc.utils import egrad

        f = lambda *x: jnp.sin(jnp.dot(jnp.hstack(x), self._w) + self._b)

        z = jnp.split(z, z.shape[1], axis=1)

        def Recurse(dark, d, dim, dCurr=0):
            if dCurr == d:
                return dark
            else:
                dark2 = egrad(dark, dim)
                dCurr += 1
                return Recurse(dark2, d, dim, dCurr=dCurr)

        dark = f
        dark2 = 1
        for dim, deriv in enumerate(d):
            dark2 *= self._c[dim] ** deriv
            dark = Recurse(dark, deriv, dim)

        return (dark(*z) * dark2).to_py()
