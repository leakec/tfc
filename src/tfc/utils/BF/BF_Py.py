import numpy as np
from abc import ABC, abstractmethod
from numpy import typing as npt
from typing import Annotated, Union
from annotated_types import Gt

uint = Annotated[int, Gt(0)]
Number = Union[int, float, complex]


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
        if len(z.shape) == 1:
            z = np.expand_dims(z, 1)
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
        if len(z.shape) == 1:
            z = np.expand_dims(z, 1)
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
        if len(z.shape) == 1:
            z = np.expand_dims(z, 1)
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
        if len(z.shape) == 1:
            z = np.expand_dims(z, 1)
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
        if len(z.shape) == 1:
            z = np.expand_dims(z, 1)
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
        if len(z.shape) == 1:
            z = np.expand_dims(z, 1)
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
