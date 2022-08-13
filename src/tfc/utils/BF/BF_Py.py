import numpy as np
from abc import ABC, abstractmethod
from numpy import typing as npt
from typing import Sequence, Annotated, Gt, Union

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
        nC: Sequence[Number],
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
        nC: Sequence[Number]
            Basis functions to be removed
        m: uint
            Number of basis functions.
        z: Number
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
        if not full:
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
                            + 2 * x * dark2[:, k - 1 : k]
                            - dark2[:, k - 2 : k - 1]
                        )
                    dCurr += 1
                    return Recurse(dark2, d, dCurr=dCurr)

            F = Recurse(F, d)
            return F
