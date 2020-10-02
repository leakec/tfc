import sys
from colorama import init as initColorama
from colorama import Fore as fg
from colorama import Style as style

from collections import OrderedDict
from functools import partial

import pickle
import numpy as onp
import matplotlib as matplotlib

# Change matplotlib backend to allow fig.show()
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from jax.config import config
config.update('jax_enable_x64', True)
import numpy as onp
import jax.numpy as np
from jax import jvp, jit, lax, jacfwd
from jax.util import safe_zip
from jax.tree_util import register_pytree_node, tree_multimap
from jax.interpreters.partial_eval import JaxprTracer

##
# This is the TFCPrint class. It is used to print text in the terminal with color.
class TFCPrint:

    def __init__(self):
        """ This function is the constructor. It initializes the colorama class. """
        initColorama()

    def Error(stringIn):
        """ This function prints errors. It prints the text in 'stringIn' in bright red and
            exits the program."""
        print(fg.RED+style.BRIGHT+stringIn)
        print(style.RESET_ALL,end="")
        sys.exit()

    def Warning(stringIn):
        """ This function prints warnings. It prints the text in 'stringIn' in bright yellow."""
        print(fg.YELLOW+style.BRIGHT+stringIn)
        print(style.RESET_ALL,end="")

def egradSimple(g,j=0):
        """ This function mimics egrad from autograd. """
        def wrapped(*args):
            tans = tuple([onp.ones(args[i].shape) if i == j else onp.zeros(args[i].shape) for i in range(len(args)) ])
            _,x_bar = jvp(g,args,tans)
            return x_bar
        return wrapped

@partial(partial, tree_multimap)
def onesRobust(val):
    """ Returns ones_like val, but can handle arrays and dictionaries. """
    return onp.ones(val.shape)

@partial(partial, tree_multimap)
def zerosRobust(val):
    """ Returns zeros_like val, but can handle arrays and dictionaries. """
    return onp.zeros(val.shape)

def egrad(g,j=0):
    """ This function mimics egrad from autograd, but can also handle dictionaries. """
    if g.__qualname__ == 'jit.<locals>.f_jitted':
        g = g.__wrapped__
    def wrapped(*args):
        tans = tuple([onesRobust(args[i]) if i == j else zerosRobust(args[i]) for i in range(len(args)) ])
        _,x_bar = jvp(g,args,tans)
        return x_bar
    return wrapped

##
# This is the TFC dictionary class. It extends an OrderedDict and
# adds a few methods that enable:
#   - Adding dictionaries with the same keys together
#   - Turning a dictionary into a 1-D array
#   - Turning a 1-D array into a dictionary
class TFCDict(OrderedDict):

    def __init__(self,*args):
        """ Initialize TfcDict using the OrderedDict method. """

        # Store dictionary and keep a record of the keys. Keys will stay in same
        # order, so that adding and subtracting is repeatable.
        super().__init__(*args)
        self._keys = list(self.keys())
        self._nKeys = len(self._keys)
        self.getSlices()

    def getSlices(self):
        """ Function that creates slices for each of the keys in the dictionary. """
        if all(isinstance(value,np.ndarray) for value in self.values()):
            arrLen = 0
            self._slices = [slice(0,0,1),]*self._nKeys
            start = 0
            stop = 0
            for k in range(self._nKeys):
                start = stop
                arrLen = self[self._keys[k]].shape[0]
                stop = start + arrLen
                self._slices[k] = slice(start,stop,1)
        else:
            self._slices = [None,]*self._nKeys

    def update(self,*args):
        """ Overload the update method to update the _keys variable as well. """
        super().update(*args)
        self._keys = list(self.keys())
        self._nKeys = len(self._keys)
        self.getSlices()

    def toArray(self):
        """ Send dictionary to a flat JAX array. """
        return np.hstack([self[self._keys[k]] for k in range(self._nKeys)])

    def toDict(self,arr):
        """ Send a flat JAX array to a TfcDict with the same keys."""
        arr = arr.flatten()
        return TFCDict(zip(self._keys,[arr[self._slices[k]] for k in range(self._nKeys)]))

    def block_until_ready(self):
        """ Mimics block_until_ready for jax arrays. Used to halt the program until the computation that created the 
            dictionary is finished. """
        self[self._keys[0]].block_until_ready()
        return self

    def __iadd__(self,o):
        """ Used to overload "+=" for TfcDict so that 2 TfcDict's can be added together."""
        if isinstance(o,dict) or (type(o) is type(self)):
            for key in self._keys:
                self[key] += o[key]
        elif isinstance(o,np.ndarray):
            o = o.flatten()
            for k in range(self._nKeys):
                self[self._keys[k]] += o[self._slices[k]]
        return self

    def __isub__(self,o):
        """ Used to overload "-=" for TfcDict so that 2 TfcDict's can be subtracted."""
        if isinstance(o,dict) or (type(o) is type(self)):
            for key in self._keys:
                self[key] -= o[key]
        elif isinstance(o,np.ndarray):
            o = o.flatten()
            for k in range(self._nKeys):
                self[self._keys[k]] -= o[self._slices[k]]
        return self

    def __add__(self,o):
        """ Used to overload "+" for TfcDict so that 2 TfcDict's can be added together."""
        out = TFCDict(self)
        if isinstance(o,dict) or (type(o) is type(self)):
            for key in self._keys:
                out[key] += o[key]
        elif isinstance(o,np.ndarray):
            o = o.flatten()
            for k in range(self._nKeys):
                out[self._keys[k]] += o[self._slices[k]]
        return out

    def __sub__(self,o):
        """ Used to overload "-" for TfcDict so that 2 TfcDict's can be subtracted."""
        out = TFCDict(self)
        if isinstance(o,dict) or (type(o) is type(self)):
            for key in self._keys:
                out[key] -= o[key]
        elif isinstance(o,np.ndarray):
            o = o.flatten()
            for k in range(self._nKeys):
                out[self._keys[k]] -= o[self._slices[k]]
        return out

# Register TFCDict as a JAX type
register_pytree_node(
  TFCDict,
  lambda x: (list(x.values()), list(x.keys())),
  lambda keys, values: TFCDict(safe_zip(keys, values)))

##
# This class is like the TFCDict class, but it handles non-flat arrays.
class TFCDictRobust(OrderedDict):
    def __init__(self,*args):
        """ Initialize TFCDictRobust using the OrderedDict method. """

        # Store dictionary and keep a record of the keys. Keys will stay in same
        # order, so that adding and subtracting is repeatable.
        super().__init__(*args)
        self._keys = list(self.keys())
        self._nKeys = len(self._keys)
        self.getSlices()

    def getSlices(self):
        """ Function that creates slices for each of the keys in the dictionary. """
        if all(isinstance(value,np.ndarray) for value in self.values()):
            arrLen = 0
            self._slices = [slice(0,0,1),]*self._nKeys
            start = 0
            stop = 0
            for k in range(self._nKeys):
                start = stop
                arrLen = self[self._keys[k]].flatten().shape[0]
                stop = start + arrLen
                self._slices[k] = slice(start,stop,1)
        else:
            self._slices = [None,]*self._nKeys

    def update(self,*args):
        """ Overload the update method to update the _keys variable as well. """
        super().update(*args)
        self._keys = list(self.keys())
        self._nKeys = len(self._keys)
        self.getSlices()

    def toArray(self):
        """ Send dictionary to a flat JAX array. """
        return np.hstack([self[self._keys[k]].flatten() for k in range(self._nKeys)])

    def toDict(self,arr):
        """ Send a flat JAX array to a TfcDict with the same keys."""
        arr = arr.flatten()
        return TFCDictRobust(zip(self._keys,[arr[self._slices[k]].reshape(self[self._keys[k]].shape) for k in range(self._nKeys)]))

    def block_until_ready(self):
        """ Mimics block_until_ready for jax arrays. Used to halt the program until the computation that created the 
            dictionary is finished. """
        self[self._keys[0]].block_until_ready()
        return self

    def __iadd__(self,o):
        """ Used to overload "+=" for TfcDict so that 2 TfcDict's can be added together."""
        if isinstance(o,dict) or (type(o) is type(self)):
            for key in self._keys:
                self[key] += o[key]
        elif isinstance(o,np.ndarray):
            o = o.flatten()
            for k in range(self._nKeys):
                self[self._keys[k]] += o[self._slices[k]].reshape(self[self._keys[k]].shape)
        return self

    def __isub__(self,o):
        """ Used to overload "-=" for TfcDict so that 2 TfcDict's can be subtracted."""
        if isinstance(o,dict) or (type(o) is type(self)):
            for key in self._keys:
                self[key] -= o[key]
        elif isinstance(o,np.ndarray):
            o = o.flatten()
            for k in range(self._nKeys):
                self[self._keys[k]] -= o[self._slices[k]].reshape(self[self._keys[k]].shape)
        return self

    def __add__(self,o):
        """ Used to overload "+" for TfcDict so that 2 TfcDict's can be added together."""
        out = TFCDictRobust(self)
        if isinstance(o,dict) or (type(o) is type(self)):
            for key in self._keys:
                out[key] += o[key]
        elif isinstance(o,np.ndarray):
            o = o.flatten()
            for k in range(self._nKeys):
                out[self._keys[k]] += o[self._slices[k]].reshape(self[self._keys[k]].shape)
        return out

    def __sub__(self,o):
        """ Used to overload "-" for TfcDict so that 2 TfcDict's can be subtracted."""
        out = TFCDictRobust(self)
        if isinstance(o,dict) or (type(o) is type(self)):
            for key in self._keys:
                out[key] -= o[key]
        elif isinstance(o,np.ndarray):
            o = o.flatten()
            for k in range(self._nKeys):
                out[self._keys[k]] -= o[self._slices[k]].reshape(self[self._keys[k]].shape)
        return out

# Register TFCDictRobust as a JAX type
register_pytree_node(
  TFCDictRobust,
  lambda x: (list(x.values()), list(x.keys())),
  lambda keys, values: TFCDictRobust(safe_zip(keys, values)))

## JIT-ed non-linear least squares.
# This function takes in an initial guess, xiInit (initial values of xi), and a residual function, res, and
# performs a nonlinear least squares to minimize the res function using the parameters
# xi. The conditions on stopping the nonlinear least-squares are:
# 1. max(abs(res)) < tol
# 2. max(abs(dxi)) < tol, where dxi is the change in xi from the last iteration.
# 3. Number of iterations > maxIter.
#
# The outputs of this function are:
# 1. xi: The values of xi that minimize the residual.
# 2. it: The number of iterations.
# 3. time: If timer = True, then the third output is the time the nonlinear least-squares took;
#          otherwise, there is no third output.
#
# The option kwarg arguments are:
# - J: User-specified jacobian that takes in argument xi. Default value is the jacobian of
#      res with respect to xi.
# - tol: Tolerance for stopping the while loop. Default is 1e-13.
# - maxIter: Maximum number of nonlinear least-squares iterations. Default is 50.
# - method: Method used to invert the matrix at each iteration. The default is pinv. The two options are,
# - cond: User specified condition function. Default is None, which results in a condition
#   that checks the three stopping conditions specified above.
# - body: User specified body function. Default is None, which results in a body function that performs
#   least-squares using the method provided and updates xi, dxi, and it.
#   * pinv: Use np.linalg.pinv
#   * lstsq: Use np.linalg.lstsq
# - timer: Setting this to True will time the non-linear least squares. Note that doing so
#          adds a slight increase in runtime. As one iteration of the non-linear least squares
#          is run first to avoid timining the JAX trace. The default is False.
# - printOut: Currently this option is not implemented. If JAX allows printing in JIT-ed functions,
#             then it will dislpay the value of max(abs(res)) at each iteration.

def NLLS(xiInit,res,*args,J=None,cond=None,body=None,tol=1e-13,maxIter=50,method='pinv',timer=False,printOut=False):

    if timer and printOut:
        TFCPrint.Warning("Warning, you have both the timer and printer on in the nonlinear least-squares.\nThe time will be longer than optimal due to the printout.")
    if printOut:
        TFCPrint.Warning("Warning, printing is not yet supported. You're going to get a garbage printout.")

    if isinstance(xiInit,TFCDict) or isinstance(xiInit,TFCDictRobust):
        dictFlag = True
    else:
        dictFlag = False

    def cond(val):
        return np.all(np.array([
                    np.max(np.abs(res(val['xi'],*args))) > tol,
                    val['it'] < maxIter,
                    np.max(np.abs(val['dxi'])) > tol]))

    if J is None:
        if dictFlag:
            if isinstance(xiInit,TFCDictRobust):
                def J(xi,*args):
                    jacob = jacfwd(res,0)(xi,*args)
                    return np.hstack([jacob[k].reshape(jacob[k].shape[0],onp.prod(onp.array(xi[k].shape))) for k in xi.keys()])
            else:
                def J(xi,*args):
                    jacob = jacfwd(res,0)(xi,*args)
                    return np.hstack([jacob[k] for k in xi.keys()])
        else:
            J = lambda xi: jacfwd(res,0)(xi,*args)

    if method == 'pinv':
        LS = lambda xi: np.dot(np.linalg.pinv(J(xi,*args)),res(xi,*args))
    elif method == 'lstsq':
        LS = lambda xi: np.linalg.lstsq(J(xi,*args),res(xi,*args),rcond=None)[0]
    else:
        TFCPrint.Error("The method entered is not valid. Please enter a valid method.")

    if body is None:
        if printOut:
            def body(val):
                val['dxi'] = LS(val['xi'])
                val['xi'] -= val['dxi']
                val['it'] += 1
                print("Iteration "+str(val['it'])+":\tMax Residual: "+str(np.max(np.abs(res(val['xi'])))))
                return val
        else:
            def body(val):
                val['dxi'] = LS(val['xi'])
                val['xi'] -= val['dxi']
                val['it'] += 1
                return val

    nlls = jit(lambda val: lax.while_loop(cond,body,val))
    if dictFlag:
        dxi = np.ones_like(xiInit.toArray())
    else:
        dxi = np.ones_like(xiInit)

    if timer:
        from time import process_time as timer
        val = {'xi':xiInit,'dxi':dxi,'it':maxIter-1}
        nlls(val)['dxi'].block_until_ready()

        val = {'xi':xiInit,'dxi':dxi,'it':0}

        start = timer()
        val = nlls(val).block_until_ready()
        val['dxi'].block_until_ready()
        stop = timer()

        return val['xi'],val['it'],stop-start
    else:
        val = {'xi':xiInit,'dxi':dxi,'it':0}
        val = nlls(val)
        return val['xi'],val['it']

class MakePlot():

    def __init__(self,xlabs,ylabs,twinYlabs=None,titles=None,zlabs=None,name='name'):
        # Set the fontsizes and family
        smallSize = 16
        mediumSize = 18
        largeSize = 18
        plt.rc('font', size=smallSize)
        plt.rc('axes', titlesize=mediumSize)
        plt.rc('axes', labelsize=largeSize)
        plt.rc('xtick', labelsize=mediumSize)
        plt.rc('ytick', labelsize=mediumSize)
        plt.rc('legend', fontsize=smallSize)
        plt.rc('figure', titlesize=largeSize)

        # Create figure and store basic labels
        self.fig = plt.figure()
        self._name = name

        # Consistify all label types
        if isinstance(xlabs,onp.ndarray):
            pass
        elif isinstance(xlabs,str):
            xlabs = onp.array([[xlabs]])
        elif isinstance(xlabs,tuple) or isinstance(xlabs,list):
            xlabs = onp.array(xlabs)
        else:
            TFCPrint.Error("The xlabels provided are not of a valid type. Please provide valid xlabels")
        if len(xlabs.shape) == 1:
            xlabs = onp.expand_dims(xlabs,1)

        if isinstance(ylabs,onp.ndarray):
            pass
        elif isinstance(ylabs,str):
            ylabs = onp.array([[ylabs]])
        elif isinstance(ylabs,tuple) or isinstance(ylabs,list):
            ylabs = onp.array(ylabs)
        else:
            TFCPrint.Error("The ylabels provided are not of a valid type. Please provide valid ylabels")
        if len(ylabs.shape) == 1:
            ylabs = onp.expand_dims(ylabs,1)

        if not zlabs is None:
            if isinstance(zlabs,onp.ndarray):
                pass
            elif isinstance(zlabs,str):
                zlabs = onp.array([[zlabs]])
            elif isinstance(zlabs,tuple) or isinstance(zlabs,list):
                zlabs = onp.array(zlabs)
            else:
                TFCPrint.Error("The zlabels provided are not of a valid type. Please provide valid zlabels")
            if len(zlabs.shape) == 1:
                zlabs = onp.expand_dims(zlabs,1)

        if titles is not None:
            if isinstance(titles,onp.ndarray):
                pass
            elif isinstance(titles,str):
                titles = onp.array([[titles]])
            elif isinstance(titles,tuple) or isinstance(titles,list):
                titles = onp.array(titles)
            else:
                TFCPrint.Error("The titles provided are not of a valid type. Please provide valid titles.")
            if len(titles.shape) == 1:
                titles = onp.expand_dims(titles,1)

        if twinYlabs is not None:
            if isinstance(twinYlabs,onp.ndarray):
                pass
            elif isinstance(twinYlabs,str):
                twinYlabs = onp.array([[twinYlabs]])
            elif isinstance(twinYlabs,tuple) or isinstance(twinYlabs,list):
                twinYlabs = onp.array(twinYlabs)
            else:
                TFCPrint.Error("The twin ylabels provided are not of a valid type. Please provide valid twin ylabels")
            if len(twinYlabs.shape) == 1:
                twinYlabs = onp.expand_dims(twinYlabs,1)


        # Create all subplots and add labels
        if zlabs is None:
            n = xlabs.shape
            self.ax = list()
            count = 0
            for j in range(n[0]):
                for k in range(n[1]):
                    if xlabs[j,k] is None:
                        continue
                    self.ax.append(self.fig.add_subplot(n[0],n[1],j*n[1]+k+1))
                    self.ax[count].set_xlabel(xlabs[j,k])
                    self.ax[count].set_ylabel(ylabs[j,k])
                    count += 1
        else:
            n = xlabs.shape
            self.ax = list()
            count = 0
            for j in range(n[0]):
                for k in range(n[1]):
                    if xlabs[j,k] is None:
                        continue
                    self.ax.append(self.fig.add_subplot(n[0],n[1],j*n[1]+k+1,projection='3d'))
                    self.ax[count].set_xlabel(xlabs[j,k])
                    self.ax[count].set_ylabel(ylabs[j,k])
                    self.ax[count].set_zlabel(zlabs[j,k])
                    count += 1
        
        if twinYlabs is not None:
            self.twinAx = list()
            count = 0
            for j in range(n[0]):
                for k in range(n[1]):
                    if xlabs[j,k] is None:
                        continue
                    self.twinAx.append(self.ax[count].twinx())
                    self.twinAx[count].set_ylabel(twinYlabs[j,k])
                    count += 1

        # Add titles if desired
        if titles is not None:
            count = 0
            for j in range(n[0]):
                for k in range(n[1]):
                    if titles[j,k] is None:
                        continue
                    self.ax[count].set_title(titles[j,k])
                    count += 1

        # Set tight layout for the figure
        self.fig.tight_layout()

    def FullScreen(self):

        # Get screensize
        import tkinter as tk
        root = tk.Tk()
        width = root.winfo_screenwidth()
        height = root.winfo_screenheight()

        # Get dpi and set new figsize
        dpi = float(self.fig.get_dpi())
        self.fig.set_size_inches(width/dpi,height/dpi)

    def PartScreen(self,width,height):

        # Get screensize
        self.fig.set_size_inches(width,height)

    def show(self):
        self.fig.show()

    def save(self,fileName,trans=True,fileType='pdf'):
        self.fig.savefig(fileName+'.'+fileType, bbox_inches='tight', pad_inches = 0, dpi = 300, format=fileType, transparent=trans)

    def savePickle(self,fileName):
        pickle.dump(self.fig,open(fileName+'.pickle','wb'))

    def saveAll(self,fileName):
        self.save(fileName)
        self.savePickle(fileName)
