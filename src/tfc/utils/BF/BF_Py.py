import numpy as np
from abc import ABC, abstractmethod
from numpy import typing as npt
from numbers import Number
from typing import Sequence, Annotated, Gt

uint = Annotated[int, Gt(0)]

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

    def __init__(self, x0: Number, xf: Number, nC: Sequence[Number], m: uint, z: Number = 0 zf: Number = float("inf")) -> None:
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

        self._nC = nC
        self._numC = len(nC)

        self._z0 = z0
        if zf == float("inf"):
            self._c = 1.0
            self._x0 = 0.0
        else:
            self._x0 = x0
            self._c = (zf-z0)/(xf-x0)

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

        z = (x-self._x0)*self._c + self._z0
        dMult = self._c**d
        F = self._Hint(z,d)*dMult
        if not full:
            F = np.delete(F, self._nC)
        return F

    @abstractmethod
    def _Hint(self, z: npt.NDArray, d: uint) -> npt.NDArray:
        pass
