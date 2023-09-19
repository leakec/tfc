import sys
from colorama import init as initColorama
from colorama import Fore as fg
from colorama import Style as style

from collections import OrderedDict
from functools import partial

from jax._src.config import config

config.update("jax_enable_x64", True)
import numpy as onp
import numpy.typing as npt
import jax.numpy as np
from jax import jvp, jit, lax, jacfwd, tree_map
from jax.extend import linear_util as lu
from jax.util import safe_zip
from jax.tree_util import register_pytree_node, tree_map
from jax._src.api_util import flatten_fun
from jax._src.tree_util import tree_flatten
from jax.core import get_aval, eval_jaxpr
from jax.interpreters.partial_eval import trace_to_jaxpr_nounits, PartialVal
from jax.experimental.host_callback import id_tap
from typing import Any, Callable, Optional, Union, cast
from .types import uint, List, Literal, Tuple, TypedDict, Path, Dict
from jaxtyping import PyTree
from typing import cast

# Types that can be added to a TFCDict
TFCDictAddable = Union[np.ndarray, Dict[Any, Any], "TFCDict"]

# Types that can be added to a TFCDictRobust
TFCDictRobustAddable = Union[np.ndarray, Dict[Any, Any], "TFCDictRobust"]


class TFCPrint:
    """
    This class is used to print to the terminal in color.
    """

    def __init__(self):
        """
        This function is the constructor. It initializes the colorama class.
        """
        initColorama()

    @staticmethod
    def Error(stringIn: str):
        """
        This function prints errors. It prints the text in 'stringIn' in bright red and
        exits the program.

        Parameters
        ----------
        stringIn : str
            error string
        """
        print(fg.RED + style.BRIGHT + stringIn)
        print(style.RESET_ALL, end="")
        sys.exit()

    @staticmethod
    def Warning(stringIn: str):
        """
        This function prints warnings. It prints the text in 'stringIn' in bright yellow.

        Parameters
        ----------
        stringIn : str
            warning string
        """
        print(fg.YELLOW + style.BRIGHT + stringIn)
        print(style.RESET_ALL, end="")


def egrad(g: Callable[..., Any], j: uint = 0):
    """
    This function mimics egrad from autograd.

    Parameters
    ----------
    g : Callable[..., Any]
        Function to take the derivative of.

    j : uint, optional
        Parameter with which to take the derivative with respect to. (Default value = 0)

    Returns
    -------
    wrapped : function
        Derivative function
    """

    def wrapped(*args: Any) -> Any:
        """
        Wrapper for derivative of g with respect to parameter number j.

        Parameters
        ----------
        *args : Any
            function arguments to g

        Returns
        -------
        x_bar: Any
            derivative of g with respect to parameter number j
        """
        tans = tuple(
            [
                onp.ones(args[i].shape) if i == j else onp.zeros(args[i].shape)
                for i in range(len(args))
            ]
        )
        _, x_bar = jvp(g, args, tans)
        return x_bar

    return wrapped


@partial(partial, tree_map)
def onesRobust(val: PyTree):
    """
    Returns ones_like val, but can handle arrays and dictionaries.

    Parameters
    ----------
    val : PyTree

    Returns
    -------
    ones_like_val : PyTree
        Pytree with the same structure as val with all elements equal to one.

    """

    # The @partial will force the PyTree to look array-like so we can use array
    # functions as done below. Using an explicit cast here so LSPs will not complain.
    val = cast(npt.NDArray, val)

    return onp.ones(val.shape, dtype=val.dtype)


@partial(partial, tree_map)
def zerosRobust(val: PyTree):
    """
    Returns zeros_like val, but can handle arrays and dictionaries.

    Parameters
    ----------
    val : PyTree

    Returns
    -------
    zeros_like_val : PyTree
        Pytree with the same structure as val with all elements equal to zero.
    """

    # The @partial will force the PyTree to look array-like so we can use array
    # functions as done below. Using an explicit cast here so LSPs will not complain.
    val = cast(npt.NDArray, val)

    return onp.zeros(val.shape, dtype=val.dtype)


def egradRobust(g: Callable[..., Any], j: uint = 0):
    """This function mimics egrad from autograd, but can also handle dictionaries.

    Parameters
    ----------
    g : function
        Function to take the derivative of.

    j : integer, optional
        Parameter with which to take the derivative with respect to. (Default value = 0)

    Returns
    -------
    wrapped : function
        Derivative function
    """
    if g.__qualname__ == "jit.<locals>.f_jitted":
        g = g.__wrapped__

    def wrapped(*args: Any) -> Any:
        """
        Wrapper for derivative of g with respect to parameter number j.

        Parameters
        ----------
        *args : iterable
            function arguments to g

        Returns
        -------
        x_bar: array-like
            derivative of g with respect to parameter number j
        """
        tans = tuple(
            [onesRobust(args[i]) if i == j else zerosRobust(args[i]) for i in range(len(args))]
        )
        _, x_bar = jvp(g, args, tans)
        return x_bar

    return wrapped


def pe(*args: Any, constant_arg_nums: List[int] = []) -> Any:
    """
    Decorator that returns a function evaluated such that the arg numbers specified in constant_arg_nums
    and all functions that utilizes only those arguments are treated as compile time constants.

    Parameters
    ----------
    *args : Any
        Arguments for the function that pe is applied to.
    constant_arg_nums : List[int], optional
        The arguments whose values and functions that depend only on these values should be
        treated as cached constants.

    Returns
    -------
    f : Any
        The new function whose constant_arg_num arguments have been removed. The jaxpr of this
        function has the constant_arg_num values and all functions that depend on those values
        cached as constants.

    Usage
    -----
    @pe(*args, constant_arg_nums=[0])
    def f(x,xi):
        # Function stuff here

    # Returns an f(xi) with x treated as constant
    """

    # Reorder to put knowns first, then unknowns
    order = [k for k in range(len(args))]
    for k in constant_arg_nums:
        order.insert(0, order.pop(k))
    reorder = np.argsort(np.array(order))
    dark = tuple(args[k] for k in order)

    # Store the removed args for later
    num_args_remove = len(constant_arg_nums)

    def wrapper(f_orig):
        if len(constant_arg_nums) > 0:
            # Reordering args so the ones to remove are given first
            # This will allow us to return a function that has completely removed those args
            # Moreover, we do it here so this reordering will be optimized by the compiler
            def f(*args):
                new_args = tuple(args[k] for k in reorder)
                return f_orig(*new_args)

            # Create the partial args needed by trace_to_jaxpr_nounits
            def get_arg(a, unknown):
                if unknown:
                    return tree_flatten(
                        (
                            tree_map(
                                lambda x: PartialVal.unknown(get_aval(x).at_least_vspace()), a
                            ),
                            {},
                        )
                    )[0]
                else:
                    return PartialVal.known(a)

            part_args = []
            for k, a in enumerate(dark):
                temp = get_arg(a, k >= num_args_remove)
                if isinstance(temp, list):
                    part_args += temp
                else:
                    part_args.append(temp)
            part_args = tuple(part_args)

            # Create jaxpr
            wrap = lu.wrap_init(f)
            _, in_tree = tree_flatten((dark, {}))
            wrap_flat, out_tree = flatten_fun(wrap, in_tree)
            jaxpr, _, const = trace_to_jaxpr_nounits(wrap_flat, part_args)

            # Create new, partially evaluated function
            if out_tree().num_leaves == 1 and out_tree().num_nodes == 1:
                # out_tree() is PyTreeDef(*), so just return the value. Since eval_jaxpr returns a list,
                # this is just value [0]
                f_removed = lambda *args: eval_jaxpr(jaxpr, const, *tree_flatten((*args, {}))[0])[0]
            else:
                # Use out_tree() to reshape the args correctly.
                f_removed = lambda *args: out_tree().unflatten(
                    eval_jaxpr(jaxpr, const, *tree_flatten((*args, {}))[0])
                )
            return f_removed
        else:
            return f_orig

    return wrapper


def pejit(*args: Any, constant_arg_nums: List[int] = [], **kwargs) -> Any:
    """
    Works like :func:`pe <tfc.utils.TFCUtils.pe>`, but also JITs the returned function. See :func:`pe <tfc.utils.TFCUtils.pe>` for more details.

    Parameters:
    -----------
    *args: Any
        Arguments for the function that pe is applied to.
    constant_arg_nums: List[int], optional
        The arguments whose values and functions that depend only on these values should be
        treated as cached constants.
    **kwargs: Any
        Keyword arguments passed on to JIT.

    Returns:
    --------
    f: Any
        The new function whose constant_arg_num arguments have been removed. The jaxpr of this
        function has the constant_arg_num values and all functions that depend on those values
        cached, and they are treated as compile time constants.
    """

    def wrap(f_orig):
        return jit(pe(*args, constant_arg_nums=constant_arg_nums)(f_orig), **kwargs)

    return wrap


class TFCDict(OrderedDict):
    """
    This is the TFC dictionary class. It extends an OrderedDict and
    adds a few methods that enable:

      - Adding dictionaries with the same keys together
      - Turning a dictionary into a 1-D array
      - Turning a 1-D array into a dictionary
    """

    def __init__(self, *args: Any):
        """
        Initialize TFCDict using the OrderedDict method.

        Parameters
        ----------
        *args : Any
            Arguments to pass to the OrderedDict
        """

        # Store dictionary and keep a record of the keys. Keys will stay in same
        # order, so that adding and subtracting is repeatable.
        super().__init__(*args)
        self._keys = list(self.keys())
        self._nKeys = len(self._keys)
        self.getSlices()

    def getSlices(self) -> None:
        """
        Function that creates slices for each of the keys in the dictionary.
        """
        if all(
            isinstance(value, np.ndarray) or isinstance(value, onp.ndarray)
            for value in self.values()
        ):
            arrLen = 0
            self._slices = [
                slice(0, 0, 1),
            ] * self._nKeys
            start = 0
            stop = 0
            for k in range(self._nKeys):
                start = stop
                arrLen = self[self._keys[k]].shape[0]
                stop = start + arrLen
                self._slices[k] = slice(start, stop, 1)
        else:
            self._slices = [
                None,
            ] * self._nKeys

    def update(self, *args: Any) -> None:
        """
        Overload the update method to update the _keys variable as well.

        Parameters
        ----------
        *args : Any
            Same as *args for the update method of ordered dict.
        """
        super().update(*args)
        self._keys = list(self.keys())
        self._nKeys = len(self._keys)
        self.getSlices()

    def toArray(self) -> np.ndarray:
        """
        Send dictionary to a flat JAX array.

        Returns
        -------
        np.ndarray
            This dictionary as a flat JAX array.
        """
        return cast(np.ndarray, np.hstack([k for k in self.values()]))

    def toDict(self, arr: np.ndarray) -> "TFCDict":
        """
        Send a flat JAX array to a TFCDict with the same keys.

        Parameters
        ----------
        arr : ndarray
            Flat JAX array to convert to TFCDict. Must have the same number of elements as total number of elements in the dictionary.

        Returns
        -------
        TFCDict
            JAX array as a TFCDict
        """
        arr = arr.flatten()
        return TFCDict(zip(self._keys, [arr[self._slices[k]] for k in range(self._nKeys)]))

    def block_until_ready(self) -> "TFCDict":
        """
        Mimics block_until_ready for jax arrays. Used to halt the program until the computation that created the
        dictionary is finished.

        Returns
        -------
        TFCDict
            This TFCDict.
        """
        self[self._keys[0]].block_until_ready()
        return self

    def __iadd__(self, o: TFCDictAddable) -> "TFCDict":
        """
        Used to overload "+=" for TFCDict so that 2 TFCDict's can be added together.

        Parameters
        ----------
        o : TFCDictAddable
            Values to add to the current dicitonary.

        Returns
        ----------
        self : TFCDict
            A copy of self after adding in the values from o.
        """
        if isinstance(o, dict) or (type(o) is type(self)):
            for key in self._keys:
                self[key] += o[key]
        elif isinstance(o, np.ndarray):
            o = o.flatten()
            for k in range(self._nKeys):
                self[self._keys[k]] += o[self._slices[k]]
        return self

    def __isub__(self, o: TFCDictAddable) -> "TFCDict":
        """
        Used to overload "-=" for TFCDict so that 2 TFCDict's can be subtracted.

        Parameters
        ----------
        o : TFCDictAddable
            Values to subtract from the current dicitonary.

        Returns
        -------
        self : TFCDict
            A copy of self after subtracting the values from o.
        """
        if isinstance(o, dict) or (type(o) is type(self)):
            for key in self._keys:
                self[key] -= o[key]
        elif isinstance(o, np.ndarray):
            o = o.flatten()
            for k in range(self._nKeys):
                self[self._keys[k]] -= o[self._slices[k]]
        return self

    def __add__(self, o: TFCDictAddable) -> "TFCDict":
        """
        Used to overload "+" for TFCDict so that 2 TFCDict's can be added together.

        Parameters
        ----------
        o : TFCDictAddable
            Values to add to the current dicitonary.

        Returns
        ----------
        out : TFCDict
            A TFCDict with values = self + o.
        """
        out = TFCDict(self)
        if isinstance(o, dict) or (type(o) is type(self)):
            for key in self._keys:
                out[key] += o[key]
        elif isinstance(o, np.ndarray):
            o = o.flatten()
            for k in range(self._nKeys):
                out[self._keys[k]] += o[self._slices[k]]
        return out

    def __sub__(self, o: TFCDictAddable) -> "TFCDict":
        """
        Used to overload "-" for TFCDict so that 2 TFCDict's can be subtracted.

        Parameters
        ----------
        o : TFCDictAddable
            Values to subtract from the current dicitonary.

        Returns
        ----------
        self : TFCDict
            A TFCDict with values = self - o.
        """
        out = TFCDict(self)
        if isinstance(o, dict) or (type(o) is type(self)):
            for key in self._keys:
                out[key] -= o[key]
        elif isinstance(o, np.ndarray):
            o = o.flatten()
            for k in range(self._nKeys):
                out[self._keys[k]] -= o[self._slices[k]]
        return out


# Register TFCDict as a JAX type
register_pytree_node(
    TFCDict,
    lambda x: (list(x.values()), list(x.keys())),
    lambda keys, values: TFCDict(safe_zip(keys, values)),
)


class TFCDictRobust(OrderedDict):
    """This class is like the :class:`TFCDict <tfc.utils.TFCUtils.TFCDict>` class, but it handles non-flat arrays."""

    def __init__(self, *args: Any):
        """
        Initialize TFCDictRobust using the OrderedDict method.

        Parameters
        ----------
        *args : Any
            Arguments to pass to the OrderedDict
        """

        # Store dictionary and keep a record of the keys. Keys will stay in same
        # order, so that adding and subtracting is repeatable.
        super().__init__(*args)
        self._keys = list(self.keys())
        self._nKeys = len(self._keys)
        self.getSlices()

    def getSlices(self) -> None:
        """
        Function that creates slices for each of the keys in the dictionary.
        """
        if all(isinstance(value, np.ndarray) for value in self.values()):
            arrLen = 0
            self._slices = [
                slice(0, 0, 1),
            ] * self._nKeys
            start = 0
            stop = 0
            for k in range(self._nKeys):
                start = stop
                arrLen = self[self._keys[k]].flatten().shape[0]
                stop = start + arrLen
                self._slices[k] = slice(start, stop, 1)
        else:
            self._slices = [
                None,
            ] * self._nKeys

    def update(self, *args: Any) -> None:
        """
        Overload the update method to update the _keys variable as well.

        Parameters
        ----------
        *args : Any
            Same as *args for the update method on ordered dict.
        """
        super().update(*args)
        self._keys = list(self.keys())
        self._nKeys = len(self._keys)
        self.getSlices()

    def toArray(self) -> np.ndarray:
        """
        Send dictionary to a flat JAX array.

        Returns
        -------
        np.ndarray
            This dictionary as a flat JAX array.
        """
        return cast(np.ndarray, np.hstack([k.flatten() for k in self.values()]))

    def toDict(self, arr: np.ndarray) -> "TFCDictRobust":
        """
        Send a flat JAX array to a TFCDictRobust with the same keys.

        Parameters
        ----------
        arr : np.ndarray
            Flat JAX array to convert to TFCDictRobust. Must have the same number of elements as total number of elements in the dictionary.

        Returns
        -------
        TFCDictRobust
            JAX array as a TFCDictRobust
        """
        arr = arr.flatten()
        return TFCDictRobust(
            zip(
                self._keys,
                [
                    arr[self._slices[k]].reshape(self[self._keys[k]].shape)
                    for k in range(self._nKeys)
                ],
            )
        )

    def block_until_ready(self) -> "TFCDictRobust":
        """
        Mimics block_until_ready for jax arrays. Used to halt the program until the computation that created the
        dictionary is finished.

        Returns
        -------
        TFCDictRobust
            This TFCDictRobust
        """
        self[self._keys[0]].block_until_ready()
        return self

    def __iadd__(self, o: TFCDictRobustAddable) -> "TFCDictRobust":
        """
        Used to overload "+=" for TFCDictRobust so that 2 TFCDictRobust's can be added together.

        Parameters
        ----------
        o : TFCDictRobustAddable
            Values to add to the current dicitonary.

        Returns
        ----------
        self : TFCDictRobust
            A copy of self after adding in the values from o.
        """
        if isinstance(o, dict) or (type(o) is type(self)):
            for key in self._keys:
                self[key] += o[key]
        elif isinstance(o, np.ndarray):
            o = o.flatten()
            for k in range(self._nKeys):
                self[self._keys[k]] += o[self._slices[k]].reshape(self[self._keys[k]].shape)
        return self

    def __isub__(self, o: TFCDictRobustAddable) -> "TFCDictRobust":
        """
        Used to overload "-=" for TFCDictRobust so that 2 TFCDictRobust's can be subtracted.

        Parameters
        ----------
        o : TFCDictRobustAddable
            Values to subtract from the current dicitonary.

        Returns
        ----------
        self : TFCDictRobust
            A copy of self after subtracting the values from o.
        """
        if isinstance(o, dict) or (type(o) is type(self)):
            for key in self._keys:
                self[key] -= o[key]
        elif isinstance(o, np.ndarray):
            o = o.flatten()
            for k in range(self._nKeys):
                self[self._keys[k]] -= o[self._slices[k]].reshape(self[self._keys[k]].shape)
        return self

    def __add__(self, o: TFCDictRobustAddable) -> "TFCDictRobust":
        """
        Used to overload "+" for TFCDictRobust so that 2 TFCDictRobust's can be added together.

        Parameters
        ----------
        o : TFCDictRobustAddable
            Values to add to the current dicitonary.

        Returns
        ----------
        out : TFCDictRobust
            A TFCDictRobust with values = self + o.
        """
        out = TFCDictRobust(self)
        if isinstance(o, dict) or (type(o) is type(self)):
            for key in self._keys:
                out[key] += o[key]
        elif isinstance(o, np.ndarray):
            o = o.flatten()
            for k in range(self._nKeys):
                out[self._keys[k]] += o[self._slices[k]].reshape(self[self._keys[k]].shape)
        return out

    def __sub__(self, o: TFCDictRobustAddable) -> "TFCDictRobust":
        """
        Used to overload "-" for TFCDictRobust so that 2 TFCDictRobust's can be subtracted.

        Parameters
        ----------
        o : TFCDictRobustAddable
            Values to subtract from the current dicitonary.

        Returns
        -------
        self : TFCDictRobust
            A TFCDictRobust with values = self - o.
        """
        out = TFCDictRobust(self)
        if isinstance(o, dict) or (type(o) is type(self)):
            for key in self._keys:
                out[key] -= o[key]
        elif isinstance(o, np.ndarray):
            o = o.flatten()
            for k in range(self._nKeys):
                out[self._keys[k]] -= o[self._slices[k]].reshape(self[self._keys[k]].shape)
        return out


# Register TFCDictRobust as a JAX type
register_pytree_node(
    TFCDictRobust,
    lambda x: (list(x.values()), list(x.keys())),
    lambda keys, values: TFCDictRobust(safe_zip(keys, values)),
)


def LS(
    zXi: PyTree,
    res: Callable,
    *args: Any,
    constant_arg_nums: List[int] = [],
    J: Optional[Callable[..., np.ndarray]] = None,
    method: Literal["pinv", "lstsq"] = "pinv",
    timer: bool = False,
    timerType: str = "process_time",
    holomorphic: bool = False,
) -> Union[PyTree, Tuple[PyTree, float]]:
    """
    JITed least squares.
    This function takes in an initial guess of zeros, zXi, and a residual function, res, and
    linear least squares to minimize the res function using the parameters
    xi.

    Parameters
    ----------
    zXi : PyTree
        Unknown parameters to be found using least-squares.

    res : Callable
        Residual function (also known as the loss function) with signature res(xi: PyTree, *args:Any, **kwargs:Any).
        Note, the first argument does not need to be named xi, this is just illustrative.

    *args : Any
        Any additional arguments taken by res other than the first PyTree argument.

    constant_arg_nums: List[int], optional
        These arguments will be removed from the residual function and treated as constant. See :func:`pejit <tfc.utils.TFCUtils.pejit>` for more details.

    J : Optional[Callable[...,np.ndarray]]
         User specified Jacobian function. If None, then the Jacobian of res with respect to xi will be calculated via automatic differentiation. (Default value = None)

    method : Literal["pinv","lstsq"], optional
         Method for least-squares inversion. (Default value = "pinv")
         * pinv - Use np.linalg.pinv
         * lstsq - Use np.linalg.lstsq

    timer : bool, optional
         Boolean that chooses whether to time the code or not. (Default value = False). Note that setting to true adds a slight increase in runtime.
         As one iteration of the non-linear least squares is run first to avoid timining the JAX trace.

    timerType : str, optional
         Any timer from the time module. (Default value = "process_time")

    holomorphic : bool, optional
         Indicates whether residual function is promised to be holomorphic. (Default value = False)

    Returns
    -------
    xi : pytree or array-like
         Unknowns that minimize res as found via least-squares. Type will be the same as zXi specified in the input.

    time : float
         Computation time as calculated by timerType specified. This output is only returned if timer = True.
    """

    if isinstance(zXi, TFCDict) or isinstance(zXi, TFCDictRobust):
        dictFlag = True
    else:
        dictFlag = False

    if J is None:
        if dictFlag:
            if isinstance(zXi, TFCDictRobust):

                def J(xi, *args):
                    jacob = jacfwd(res, 0, holomorphic=holomorphic)(xi, *args)
                    return np.hstack(
                        [
                            jacob[k].reshape(jacob[k].shape[0], onp.prod(onp.array(xi[k].shape)))
                            for k in xi.keys()
                        ]
                    )

            else:

                def J(xi, *args):
                    jacob = jacfwd(res, 0, holomorphic=holomorphic)(xi, *args)
                    return np.hstack([jacob[k] for k in xi.keys()])

        else:
            J = lambda xi, *args: jacfwd(res, 0, holomorphic=holomorphic)(xi, *args)

    if method == "pinv":
        ls = lambda xi, *args: np.dot(np.linalg.pinv(J(xi, *args)), -res(xi, *args))
    elif method == "lstsq":
        ls = lambda xi, *args: np.linalg.lstsq(J(xi, *args), -res(xi, *args), rcond=None)[0]
    else:
        TFCPrint.Error("The method entered is not valid. Please enter a valid method.")

    if constant_arg_nums:
        # Make arguments constant if desired
        ls = pe(zXi, *args, constant_arg_nums=constant_arg_nums)(ls)

        args: List[Any] = list(args)
        constant_arg_nums.sort()
        constant_arg_nums.reverse()
        for k in constant_arg_nums:
            args.pop(k - 1)

    ls = jit(ls)
    zXi = zerosRobust(zXi)

    if timer:
        import time

        timer_f: Callable[[], float] = getattr(time, timerType)
        ls(zXi, *args).block_until_ready()

        start = timer_f()
        xi = ls(zXi, *args).block_until_ready()
        stop = timer_f()
        zXi += xi

        return zXi, stop - start
    else:
        zXi += ls(zXi, *args)
        return zXi


class LsClass:
    """
    JITed linear least-squares class.
    Like the :func:`LS <tfc.utils.TFCUtils.LS>` function, but it is in class form so that the run methd can be called multiple times without re-JITing.
    See :func:`LS <tfc.utils.TFCUtils.LS>` for more details.
    """

    def __init__(
        self,
        zXi: PyTree,
        res: Callable,
        *args: Any,
        constant_arg_nums: List[int] = [],
        J: Optional[Callable[..., np.ndarray]] = None,
        method: Literal["pinv", "lstsq"] = "pinv",
        timer: bool = False,
        timerType: str = "process_time",
        holomorphic: bool = False,
    ) -> None:
        """
        Initialization function. Creates the JIT-ed least-squares function.

        Parameters
        ----------
        zXi : PyTree
            Unknown parameters to be found using least-squares.

        res : Callable
            Residual function (also known as the loss function) with signature res(xi: PyTree, *args:Any, **kwargs:Any).
            Note, the first argument does not need to be named xi, this is just illustrative.

        *args : Any
            Any additional arguments taken by res other than the first PyTree argument.

        J : Optional[Callable[...,np.ndarray]]
             User specified Jacobian function. If None, then the Jacobian of res with respect to xi will be calculated via automatic differentiation. (Default value = None)

        constant_arg_nums: List[int], optional
            These arguments will be removed from the residual function and treated as constant. See :func:`pejit <tfc.utils.TFCUtils.pejit>` for more details.

        method : Literal["pinv","lstsq"], optional
             Method for least-squares inversion. (Default value = "pinv")
             * pinv - Use np.linalg.pinv
             * lstsq - Use np.linalg.lstsq

        timer : bool, optional
             Boolean that chooses whether to time the code or not. (Default value = False). Note that setting to true adds a slight increase in runtime.
             As one iteration of the non-linear least squares is run first to avoid timining the JAX trace.

        timerType : str, optional
             Any timer from the time module. (Default value = "process_time")

        holomorphic : bool, optional
             Indicates whether residual function is promised to be holomorphic. (Default value = False)
        """

        self.timerType = timerType
        self.timer = timer
        self.holomorphic = holomorphic

        if isinstance(zXi, TFCDict) or isinstance(zXi, TFCDictRobust):
            dictFlag = True
        else:
            dictFlag = False

        if J is None:
            if dictFlag:
                if isinstance(zXi, TFCDictRobust):

                    def J(xi, *args):
                        jacob = jacfwd(res, 0, holomorphic=self.holomorphic)(xi, *args)
                        return np.hstack(
                            [
                                jacob[k].reshape(
                                    jacob[k].shape[0], onp.prod(onp.array(xi[k].shape))
                                )
                                for k in xi.keys()
                            ]
                        )

                else:

                    def J(xi, *args):
                        jacob = jacfwd(res, 0, holomorphic=self.holomorphic)(xi, *args)
                        return np.hstack([jacob[k] for k in xi.keys()])

            else:
                J = lambda xi, *args: jacfwd(res, 0, holomorphic=self.holomorphic)(xi, *args)

        if method == "pinv":
            ls = lambda xi, *args: np.dot(np.linalg.pinv(J(xi, *args)), -res(xi, *args))
        elif method == "lstsq":
            ls = lambda xi, *args: np.linalg.lstsq(J(xi, *args), -res(xi, *args), rcond=None)[0]

        else:
            TFCPrint.Error("The method entered is not valid. Please enter a valid method.")

        if constant_arg_nums:
            # Make arguments constant if desired
            ls = pe(zXi, *args, constant_arg_nums=constant_arg_nums)(ls)

            args: List[Any] = list(args)
            constant_arg_nums.sort()
            constant_arg_nums.reverse()
            for k in constant_arg_nums:
                args.pop(k - 1)

        self._ls = jit(ls)

        self._compiled = False

    def run(self, zXi: PyTree, *args: Any) -> Union[PyTree, Tuple[PyTree, float]]:
        """
        Runs the JIT-ed least-squares function and times it if desired.

        Parameters
        ----------
        zXi : PyTree
            Unknown parameters to be found using least-squares.

        *args : Any
            Any additional arguments taken by res other than xi.

        Returns
        -------
        xi : PyTree
             Unknowns that minimize res as found via least-squares. Type will be the same as zXi specified in the input.

        time : float, optional
             Computation time as calculated by timerType specified. This output is only returned if timer = True.

        """

        if self.timer:
            import time

            timer = getattr(time, self.timerType)

            if not self._compiled:
                self._ls(zXi, *args).block_until_ready()
                self._compiled = True

            start = timer()
            xi = self._ls(zXi, *args).block_until_ready()
            stop = timer()
            zXi += xi

            return zXi, stop - start

        else:
            zXi += ls(zXi, *args)

            self._compiled = True

            return zXi


def nlls_printout(arg: Any, transforms: Any, *, end: str = "\n", **kwargs: Any) -> None:
    print("Iteration: {0}\tmax(abs(res)): {1}".format(*arg), end=end)
    return


def nlls_id_print(it: int, x, end: str = "\n"):
    printer = partial(nlls_printout, end=end)
    return id_tap(printer, (it, x))


def NLLS(
    xiInit: PyTree,
    res: Callable,
    *args: Any,
    constant_arg_nums: List[int] = [],
    J: Optional[Callable[..., np.ndarray]] = None,
    cond: Optional[Callable[[PyTree], bool]] = None,
    body: Optional[Callable[[PyTree], PyTree]] = None,
    tol: float = 1e-13,
    maxIter: uint = 50,
    method: Literal["pinv", "lstsq"] = "pinv",
    timer: bool = False,
    printOut: bool = False,
    printOutEnd: str = "\n",
    timerType: str = "process_time",
    holomorphic: bool = False,
) -> Union[Tuple[PyTree, int], Tuple[PyTree, int, float]]:
    """
    JIT-ed non-linear least squares.
    This function takes in an initial guess, xiInit (initial values of xi), and a residual function, res, and
    performs a nonlinear least squares to minimize the res function using the parameters
    xi. The conditions on terminating the nonlinear least-squares are:
    1. max(abs(res)) < tol
    2. max(abs(dxi)) < tol, where dxi is the change in xi from the last iteration.
    3. Number of iterations > maxIter.

    Parameters
    ----------
    xiInit : pytree or array-like
        Initial guess for the unkown parameters.

    res : function
        Residual function (also known as the loss function) with signature res(xi,*args).

    *args : iterable
        Any additional arguments taken by res other than xi.

    constant_arg_nums: List[int], optional
        These arguments will be removed from the residual function and treated as constant. See :func:`pejit <tfc.utils.TFCUtils.pejit>` for more details.

    J : function
         User specified Jacobian. If None, then the Jacobian of res with respect to xi will be calculated via automatic differentiation. (Default value = None)

    cond : Optional[Callable[[PyTree],bool]]
         User specified condition function. If None, then the default cond function is used which checks the three termination criteria
         provided in the class description. (Default value = None)

    body : Optional[Callable[[PyTree],PyTree]]
         User specified while-loop body function. If None, then use the default body function which updates xi using a NLLS interation and the method provided.
         (Default value = None)

    tol : float
         Tolerance used in the default termination criteria: see class description for more details. (Default value = 1e-13)

    maxIter : int, optional
         Maximum number of iterations. (Default value = 50)

    method : Literal["pinv","lstsq"], optional
         Method for least-squares inversion. (Default value = "pinv")
         * pinv - Use np.linalg.pinv
         * lstsq - Use np.linalg.lstsq

    timer : bool, optional
         Boolean that chooses whether to time the code or not. (Default value = False). Note that setting to true adds a slight increase in runtime.
         As one iteration of the non-linear least squares is run first to avoid timining the JAX trace.

    printOut : bool, optional
         Controls whether the NLLS prints out information each interaton or not. The printout consists of the iteration and max(abs(res)) at each iteration. (Default value = False)

    printOutEnd : str, optional
         Value of keyword argument end passed to the print statement used in printOut. (Default value = "\\\\n")

    timerType : str, optional
         Any timer from the time module. (Default value = "process_time")

    holomorphic : bool, optional
         Indicates whether residual function is promised to be holomorphic. (Default value = False)

    Returns
    -------
    xi : PyTree
         Unknowns that minimize res as found via least-squares. Type will be the same as zXi specified in the input.

    it : int
         Number of NLLS iterations performed.

    time : float
         Computation time as calculated by timerType specified. This output is only returned if timer = True.
    """

    if timer and printOut:
        TFCPrint.Warning(
            "Warning, you have both the timer and printer on in the nonlinear least-squares.\nThe time will be longer than optimal due to the printout."
        )

    if isinstance(xiInit, TFCDict) or isinstance(xiInit, TFCDictRobust):
        dictFlag = True
    else:
        dictFlag = False

    def cond(val):
        return np.all(
            np.array(
                [
                    np.max(np.abs(res(val["xi"], *val["args"]))) > tol,
                    val["it"] < maxIter,
                    np.max(np.abs(val["dxi"])) > tol,
                ]
            )
        )

    if J is None:
        if dictFlag:
            if isinstance(xiInit, TFCDictRobust):

                def J(xi, *args):
                    jacob = jacfwd(res, 0, holomorphic=holomorphic)(xi, *args)
                    return np.hstack(
                        [
                            jacob[k].reshape(jacob[k].shape[0], onp.prod(onp.array(xi[k].shape)))
                            for k in xi.keys()
                        ]
                    )

            else:

                def J(xi, *args):
                    jacob = jacfwd(res, 0, holomorphic=holomorphic)(xi, *args)
                    return np.hstack([jacob[k] for k in xi.keys()])

        else:
            J = lambda xi, *args: jacfwd(res, 0, holomorphic=holomorphic)(xi, *args)

    if method == "pinv":
        LS = lambda xi, *args: np.dot(np.linalg.pinv(J(xi, *args)), res(xi, *args))
    elif method == "lstsq":
        LS = lambda xi, *args: np.linalg.lstsq(J(xi, *args), res(xi, *args), rcond=None)[0]
    else:
        TFCPrint.Error("The method entered is not valid. Please enter a valid method.")

    if constant_arg_nums:
        # Make arguments constant if desired
        LS = pe(xiInit, *args, constant_arg_nums=constant_arg_nums)(LS)
        res = pe(xiInit, *args, constant_arg_nums=constant_arg_nums)(res)

        args: List[Any] = list(args)
        constant_arg_nums.sort()
        constant_arg_nums.reverse()
        for k in constant_arg_nums:
            args.pop(k - 1)

    if body is None:
        if printOut:

            def body(val):
                val["dxi"] = LS(val["xi"], *val["args"])
                val["xi"] -= val["dxi"]
                nlls_id_print(
                    val["it"], np.max(np.abs(res(val["xi"], *val["args"]))), end=printOutEnd
                )
                val["it"] += 1
                return val

        else:

            def body(val):
                val["dxi"] = LS(val["xi"], *val["args"])
                val["xi"] -= val["dxi"]
                val["it"] += 1
                return val

    nlls = jit(lambda val: lax.while_loop(cond, body, val))

    if dictFlag:
        dxi = np.ones_like(cast(Union[TFCDict, TFCDictRobust], xiInit).toArray())
    else:
        dxi = np.ones_like(xiInit)

    if timer:
        import time

        timer_f: Callable[[], float] = getattr(time, timerType)
        val = {"xi": xiInit, "dxi": dxi, "it": maxIter - 1, "args": args}
        nlls(val)["dxi"].block_until_ready()

        val = {"xi": xiInit, "dxi": dxi, "it": 0, "args": args}

        start = timer_f()
        val = nlls(val)
        val["dxi"].block_until_ready()
        stop = timer_f()

        return val["xi"], val["it"], stop - start
    else:
        val = {"xi": xiInit, "dxi": dxi, "it": 0, "args": args}
        val = nlls(val)
        return val["xi"], val["it"]


class NllsClass:
    """
    JITed nonlinear least squares class.
    Like the :func:`NLLS <tfc.utils.TFCUtils.NLLS>` function, but it is in class form so that the run methd can be called multiple times without re-JITing
    """

    def __init__(
        self,
        xiInit: PyTree,
        res: Callable,
        *args: Any,
        constant_arg_nums: List[int] = [],
        J: Optional[Callable[..., np.ndarray]] = None,
        cond: Optional[Callable[[PyTree], bool]] = None,
        body: Optional[Callable[[PyTree], PyTree]] = None,
        tol: float = 1e-13,
        maxIter: uint = 50,
        method: Literal["pinv", "lstsq"] = "pinv",
        timer: bool = False,
        printOut: bool = False,
        printOutEnd: str = "\n",
        timerType: str = "process_time",
        holomorphic: bool = False,
    ) -> None:
        """
        Initialization function. Creates the JIT-ed nonlinear least-squares function.

        Parameters
        ----------
        xiInit : pytree or array-like
            Initial guess for the unkown parameters.

        res : function
            Residual function (also known as the loss function) with signature res(xi,*args).

        *args : iterable
            Any additional arguments taken by res other than xi.

        constant_arg_nums: List[int], optional
            These arguments will be removed from the residual function and treated as constant. See :func:`pejit <tfc.utils.TFCUtils.pejit>` for more details.

        J : function
             User specified Jacobian. If None, then the Jacobian of res with respect to xi will be calculated via automatic differentiation. (Default value = None)

        cond : Optional[Callable[[PyTree],bool]]
             User specified condition function. If None, then the default cond function is used which checks the three termination criteria
             provided in the class description. (Default value = None)

        body : Optional[Callable[[PyTree],PyTree]]
             User specified while-loop body function. If None, then use the default body function which updates xi using a NLLS interation and the method provided.
             (Default value = None)

        tol : float
             Tolerance used in the default termination criteria: see class description for more details. (Default value = 1e-13)

        maxIter : int, optional
             Maximum number of iterations. (Default value = 50)

        method : Literal["pinv","lstsq"], optional
             Method for least-squares inversion. (Default value = "pinv")
             * pinv - Use np.linalg.pinv
             * lstsq - Use np.linalg.lstsq

        timer : bool, optional
             Boolean that chooses whether to time the code or not. (Default value = False). Note that setting to true adds a slight increase in runtime.
             As one iteration of the non-linear least squares is run first to avoid timining the JAX trace.

        printOut : bool, optional
             Controls whether the NLLS prints out information each interaton or not. The printout consists of the iteration and max(abs(res)) at each iteration. (Default value = False)

        printOutEnd : str, optional
             Value of keyword argument end passed to the print statement used in printOut. (Default value = "\\\\n")

        timerType : str, optional
             Any timer from the time module. (Default value = "process_time")

        holomorphic : bool, optional
             Indicates whether residual function is promised to be holomorphic. (Default value = False)
        """

        self.timerType = timerType
        self.timer = timer
        self._maxIter = maxIter
        self.holomorphic = holomorphic

        if timer and printOut:
            TFCPrint.Warning(
                "Warning, you have both the timer and printer on in the nonlinear least-squares.\nThe time will be longer than optimal due to the printout."
            )

        if isinstance(xiInit, TFCDict) or isinstance(xiInit, TFCDictRobust):
            self._dictFlag = True
        else:
            self._dictFlag = False

        def cond(val):
            return np.all(
                np.array(
                    [
                        np.max(np.abs(res(val["xi"], *val["args"]))) > tol,
                        val["it"] < maxIter,
                        np.max(np.abs(val["dxi"])) > tol,
                    ]
                )
            )

        if J is None:
            if self._dictFlag:
                if isinstance(xiInit, TFCDictRobust):

                    def J(xi, *args):
                        jacob = jacfwd(res, 0, holomorphic=self.holomorphic)(xi, *args)
                        return np.hstack(
                            [
                                jacob[k].reshape(
                                    jacob[k].shape[0], onp.prod(onp.array(xi[k].shape))
                                )
                                for k in xi.keys()
                            ]
                        )

                else:

                    def J(xi, *args):
                        jacob = jacfwd(res, 0, holomorphic=self.holomorphic)(xi, *args)
                        return np.hstack([jacob[k] for k in xi.keys()])

            else:
                J = lambda xi, *args: jacfwd(res, 0, holomorphic=self.holomorphic)(xi, *args)

        if method == "pinv":
            LS = lambda xi, *args: np.dot(np.linalg.pinv(J(xi, *args)), res(xi, *args))
        elif method == "lstsq":
            LS = lambda xi, *args: np.linalg.lstsq(J(xi, *args), res(xi, *args), rcond=None)[0]
        else:
            TFCPrint.Error("The method entered is not valid. Please enter a valid method.")

        if constant_arg_nums:
            # Make arguments constant if desired
            LS = pe(xiInit, *args, constant_arg_nums=constant_arg_nums)(LS)
            res = pe(xiInit, *args, constant_arg_nums=constant_arg_nums)(res)

            args: List[Any] = list(args)
            constant_arg_nums.sort()
            constant_arg_nums.reverse()
            for k in constant_arg_nums:
                args.pop(k - 1)

        if body is None:
            if printOut:

                def body(val):
                    val["dxi"] = LS(val["xi"], *val["args"])
                    val["xi"] -= val["dxi"]
                    nlls_id_print(
                        val["it"], np.max(np.abs(res(val["xi"], *val["args"]))), end=printOutEnd
                    )
                    val["it"] += 1
                    return val

            else:

                def body(val):
                    val["dxi"] = LS(val["xi"], *val["args"])
                    val["xi"] -= val["dxi"]
                    val["it"] += 1
                    return val

        self._nlls = jit(lambda val: lax.while_loop(cond, body, val))
        self._compiled = False

    def run(
        self, xiInit: PyTree, *args: Any
    ) -> Union[Tuple[PyTree, int], Tuple[PyTree, int, float]]:
        """Runs the JIT-ed nonlinear least-squares function and times it if desired.

        Parameters
        ----------
        xiInit : PyTree
            Initial guess for the unkown parameters.

        *args : Any
            Any additional arguments taken by res other than xi.

        Returns
        -------
        xi : PyTree
             Unknowns that minimize res as found via least-squares. Type will be the same as zXi specified in the input.

        it : int
             Number of NLLS iterations performed..

        time : float
             Computation time as calculated by timerType specified. This output is only returned if timer = True.

        """

        if self._dictFlag:
            dxi = np.ones_like(cast(Union[TFCDict, TFCDictRobust], xiInit).toArray())
        else:
            dxi = np.ones_like(xiInit)

        if self.timer:
            import time

            timer_f: Callable[[], float] = getattr(time, self.timerType)

            if not self._compiled:
                val = {"xi": xiInit, "dxi": dxi, "it": self._maxIter - 1, "args": args}
                self._nlls(val)["dxi"].block_until_ready()
                self._compiled = True

            val = {"xi": xiInit, "dxi": dxi, "it": 0, "args": args}

            start = timer_f()
            val = self._nlls(val)
            val["dxi"].block_until_ready()
            stop = timer_f()

            return val["xi"], val["it"], stop - start

        else:
            val = {"xi": xiInit, "dxi": dxi, "it": 0, "args": args}
            val = self._nlls(val)

            self._compiled = True

            return val["xi"], val["it"]


class ComponentConstraintDict(TypedDict):
    name: str
    node0: str
    node1: str


class ComponentConstraintGraph:
    """
    Creates a graph of all valid ways in which component constraints can be embedded.
    """

    def __init__(self, N: List[str], E: List[ComponentConstraintDict]) -> None:
        """
        Class constructor.

        Parameters
        ----------
        N : List[str]
            A list of strings that specify the node names. These node names typically coincide with
            the names of the dependent variables.
        E : List[ComponentConstraintDict]
            The ComponentConstraintDict is a dictionary with the following fields:
            * name - Name of the component constraint.
            * node0 - The name of one of the nodes that makes up the component constraint.  Must correspond with an element of the list given in N.
            * node1 - The name of one of the nodes that makes up the component constraint.  Must correspond with an element of the list given in N.
        """

        # Check that all edges are connected to valid nodes
        self.nNodes = len(N)
        self.nEdges = len(E)
        for k in range(self.nEdges):
            if not (E[k]["node0"] in N and E[k]["node1"] in N):
                TFCPrint.Error(
                    "Error either "
                    + E[k]["node0"]
                    + " or "
                    + E[k]["node1"]
                    + " is not a valid node. Make sure they appear in the nodes list."
                )

        # Create all possible source/target pairs. This tells whether node0 is the target or source, node1 will be the opposite.
        import itertools

        self.targets = list(itertools.product([0, 1], repeat=self.nEdges))

        # Find all targets that are valid trees
        self.goodTargets = []
        for j in range(len(self.targets)):
            adj = onp.zeros((self.nNodes, self.nNodes))
            for k in range(self.nNodes):
                kNode = N[k]
                for g in range(self.nEdges):
                    if E[g]["node0"] == kNode:
                        if self.targets[j][g]:
                            adj[N.index(E[g]["node1"]), N.index(E[g]["node0"])] = 1.0
                    elif E[g]["node1"] == kNode:
                        if not self.targets[j][g]:
                            adj[N.index(E[g]["node0"]), N.index(E[g]["node1"])] = 1.0
            if np.all(np.linalg.eigvals(adj) == 0.0):
                self.goodTargets.append(j)

        # Save nodes and edges for use later
        self.N = N
        self.E = E

    def SaveGraphs(self, outputDir: Path, allGraphs: bool = False, savePDFs: bool = False) -> None:
        """
        Saves the graphs.
        The graphs are saved in a clickable HTML structure. They can also be saved as PDFs.

        Parameters
        ----------
        outputDir : Path
            Output directory to save in.

        allGraphs : bool, optional
             Boolean that conrols whether all graphs are saved or just valid graphs. (Default value = False)

        savePDFs : bool, optional
             Boolean that controls whether the graphs are also saved as PDFs. (Default value = False)
        """
        import os
        from .Html import HTML, Dot

        if allGraphs:
            targets = self.targets
        else:
            targets = [self.targets[k] for k in self.goodTargets]

        n = len(targets)

        #: Create the main dot file
        mainDot = Dot(os.path.join(outputDir, "dotFiles", "main"), "main")
        mainDot.dot.node_attr.update(shape="box")
        mainDot.dot.edge_attr.update(style="invis")
        treeCnt = 0
        for j in range(int(np.ceil(n / 5))):
            if j != 0:
                mainDot.dot.edge("tree" + str((j - 1) * 5), "tree" + str(j * 5))
            with mainDot.dot.subgraph(name="subgraph" + str(j)) as c:
                c.attr(rank="same")
                for k in range(min(5, n - j * 5)):
                    c.node(
                        "tree" + str(treeCnt),
                        "Tree " + str(treeCnt),
                        href=os.path.join("htmlFiles", "tree" + str(treeCnt) + ".html"),
                    )
                    treeCnt += 1

        mainDot.Render()

        #: Create the main file HTML
        mainHtml = HTML(os.path.join(outputDir, "main.html"))
        with mainHtml.tag("html"):
            with mainHtml.tag("body"):
                with mainHtml.tag("style"):
                    mainHtml.doc.asis(mainHtml.centerClass)
                mainHtml.doc.stag(
                    "img", src=os.path.join("dotFiles", "main.svg"), usemap="#main", klass="center"
                )
                mainHtml.doc.asis(
                    mainHtml.ReadFile(os.path.join(outputDir, "dotFiles", "main.cmapx"))
                )
        mainHtml.WriteFile()

        #: Create the tree dot files
        for k in range(n):
            treeDot = Dot(os.path.join(outputDir, "dotFiles", "tree" + str(k)), "tree" + str(k))
            treeDot.dot.attr(bgcolor="transparent")
            treeDot.dot.node_attr.update(shape="box")
            for j in range(self.nNodes):
                treeDot.dot.node(self.N[j], self.N[j])
            for j in range(self.nEdges):
                if not targets[k][j]:
                    treeDot.dot.edge(
                        self.E[j]["node0"], self.E[j]["node1"], label=self.E[j]["name"]
                    )
                else:
                    treeDot.dot.edge(
                        self.E[j]["node1"], self.E[j]["node0"], label=self.E[j]["name"]
                    )

            if savePDFs:
                treeDot.Render(formats=["cmapx", "svg", "pdf"])
            else:
                treeDot.Render()

        #: Create the tree HTML files
        for k in range(n):
            treeHtml = HTML(os.path.join(outputDir, "htmlFiles", "tree" + str(k) + ".html"))
            with treeHtml.tag("html"):
                with treeHtml.tag("body"):
                    with treeHtml.tag("style"):
                        treeHtml.doc.asis(treeHtml.centerClass)
                    treeHtml.doc.stag(
                        "img",
                        src=os.path.join("..", "dotFiles", "tree" + str(k) + ".svg"),
                        usemap="#tree" + str(k),
                        klass="center",
                    )
                    treeHtml.doc.asis(
                        treeHtml.ReadFile(
                            os.path.join(outputDir, "dotFiles", "tree" + str(k) + ".cmapx")
                        )
                    )
            treeHtml.WriteFile()


def ScaledQrLs(A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """This function performs least-squares using a scaled QR method.

    Parameters
    ----------
    A : np.ndarray
        A matrix in A*x = B.

    B : np.ndarray
        B matrix in A*x = B.

    Returns
    -------
    x : np.ndarray
        Solution to A*x = B solved using a scaled QR method.

    cn : np.ndarray
        Condition number.
    """
    S = 1.0 / np.sqrt(np.sum(A * A, 0))
    S = np.reshape(S, (A.shape[1],))
    q, r = np.linalg.qr(A.dot(np.diag(S)))
    x: np.ndarray = S * np.linalg.multi_dot([_MatPinv(r), q.T, B])
    cn: np.ndarray = cast(np.ndarray, np.linalg.cond(r))
    return x, cn


def _MatPinv(A: np.ndarray) -> np.ndarray:
    """This function is used to better replicate MATLAB's pseudo-inverse.

    Parameters
    ----------
    A : np.ndarray
        Matrix to be inverted.

    Returns
    -------
    Ainv : np.ndarray
        Inverse of A.
    """
    rcond = onp.max(A.shape) * onp.spacing(np.linalg.norm(A, ord=2))
    return np.linalg.pinv(A, rcond=rcond)


def step(x: np.ndarray) -> np.ndarray:
    """
    This is the unit step function, but the deriative is defined and equal to 0 at every point.

    Parameters
    ----------
    x : np.ndarray
        Array to apply step to.


    Returns
    -------
    step_x : np.ndarray
        step(x)
    """
    return np.heaviside(x, 0)


def criuCheckpoint(dir: str = "criu_checkpoint", user_mode: bool = False):
    """
    Use CRIU to create a checkpoint of your program.
    WARNING: You must ensure no external sockets are used, i.e., via matplotlib.
    WARNING: GPU memory cannot be mapped for all GPUs.
    WARNING: user_mode is a work in progress and is not full featured yet.

    Parameters
    ----------
    dir : str
        Directory where the CRIU checkpoint will be created.
    user_mode : bool
        Whether to use user mode or not. (Default value = False)
    """

    import os
    from pathlib import Path

    path = Path(dir)
    pid = os.getpid()

    # Create path if it does not exist
    if not os.path.exists(path):
        os.mkdir(path)

    # Run the command and print how to restart
    if user_mode:
        os.system(
            f"(criu dump --unpriviledged -t {pid} -D {str(path.absolute())} --shell-job --leave-running &) &"
        )
        print(
            f"Creating a checkpoint. To restart run: sudo criu restore -D {str(path.absolute())} --shell-job"
        )
    else:
        os.system(
            f"(sudo criu dump -t {pid} -D {str(path.absolute())} --shell-job --leave-running &) &"
        )
        print(
            f"Creating a checkpoint. To restart run: sudo criu restore -D {str(path.absolute())} --shell-job"
        )
