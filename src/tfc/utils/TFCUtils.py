import sys
from colorama import init as initColorama
from colorama import Fore as fg
from colorama import Style as style

from collections import OrderedDict
from functools import partial

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

    @staticmethod
    def Error(stringIn):
        """ This function prints errors. It prints the text in 'stringIn' in bright red and
            exits the program."""
        print(fg.RED+style.BRIGHT+stringIn)
        print(style.RESET_ALL,end="")
        sys.exit()

    @staticmethod
    def Warning(stringIn):
        """ This function prints warnings. It prints the text in 'stringIn' in bright yellow."""
        print(fg.YELLOW+style.BRIGHT+stringIn)
        print(style.RESET_ALL,end="")

def egrad(g,j=0):
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

def egradRobust(g,j=0):
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

def NLLS(xiInit,res,*args,J=None,cond=None,body=None,tol=1e-13,maxIter=50,method='pinv',timer=False,printOut=False,timerType='process_time'):

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
                    np.max(np.abs(res(val['xi'],*val['args']))) > tol,
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
            J = lambda xi,*args: jacfwd(res,0)(xi,*args)

    if method == 'pinv':
        LS = lambda xi,*args: np.dot(np.linalg.pinv(J(xi,*args)),res(xi,*args))
    elif method == 'lstsq':
        LS = lambda xi,*args: np.linalg.lstsq(J(xi,*args),res(xi,*args),rcond=None)[0]
    else:
        TFCPrint.Error("The method entered is not valid. Please enter a valid method.")

    if body is None:
        if printOut:
            def body(val):
                val['dxi'] = LS(val['xi'],*val['args'])
                val['xi'] -= val['dxi']
                val['it'] += 1
                print("Iteration "+str(val['it'])+":\tMax Residual: "+str(np.max(np.abs(res(val['xi'])))))
                return val
        else:
            def body(val):
                val['dxi'] = LS(val['xi'],*val['args'])
                val['xi'] -= val['dxi']
                val['it'] += 1
                return val

    nlls = jit(lambda val: lax.while_loop(cond,body,val))
    if dictFlag:
        dxi = np.ones_like(xiInit.toArray())
    else:
        dxi = np.ones_like(xiInit)

    if timer:
        import time
        timer = getattr(time,timerType)
        val = {'xi':xiInit,'dxi':dxi,'it':maxIter-1,'args':args}
        nlls(val)['dxi'].block_until_ready()

        val = {'xi':xiInit,'dxi':dxi,'it':0,'args':args}

        start = timer()
        val = nlls(val)
        val['dxi'].block_until_ready()
        stop = timer()

        return val['xi'],val['it'],stop-start
    else:
        val = {'xi':xiInit,'dxi':dxi,'it':0,'args':args}
        val = nlls(val)
        return val['xi'],val['it']

## JIT-ed non-linear least squares class.
# Like the NLLS function, but it is in class form so that the run methd can be called multiple times w/o re-JITing 

class NllsClass:

    def __init__(self,xiInit,res,J=None,cond=None,body=None,tol=1e-13,maxIter=50,method='pinv',timer=False,printOut=False,timerType='process_time'):

        self.timerType = timerType
        self.timer = timer
        self._maxIter = maxIter

        if timer and printOut:
            TFCPrint.Warning("Warning, you have both the timer and printer on in the nonlinear least-squares.\nThe time will be longer than optimal due to the printout.")
        if printOut:
            TFCPrint.Warning("Warning, printing is not yet supported. You're going to get a garbage printout.")

        if isinstance(xiInit,TFCDict) or isinstance(xiInit,TFCDictRobust):
            self._dictFlag = True
        else:
            self._dictFlag = False

        def cond(val):
            return np.all(np.array([
                        np.max(np.abs(res(val['xi'],*val['args']))) > tol,
                        val['it'] < maxIter,
                        np.max(np.abs(val['dxi'])) > tol]))

        if J is None:
            if self._dictFlag:
                if isinstance(xiInit,TFCDictRobust):
                    def J(xi,*args):
                        jacob = jacfwd(res,0)(xi,*args)
                        return np.hstack([jacob[k].reshape(jacob[k].shape[0],onp.prod(onp.array(xi[k].shape))) for k in xi.keys()])
                else:
                    def J(xi,*args):
                        jacob = jacfwd(res,0)(xi,*args)
                        return np.hstack([jacob[k] for k in xi.keys()])
            else:
                J = lambda xi,*args: jacfwd(res,0)(xi,*args)

        if method == 'pinv':
            LS = lambda xi,*args: np.dot(np.linalg.pinv(J(xi,*args)),res(xi,*args))
        elif method == 'lstsq':
            LS = lambda xi,*args: np.linalg.lstsq(J(xi,*args),res(xi,*args),rcond=None)[0]
        else:
            TFCPrint.Error("The method entered is not valid. Please enter a valid method.")

        if body is None:
            if printOut:
                def body(val):
                    val['dxi'] = LS(val['xi'],*val['args'])
                    val['xi'] -= val['dxi']
                    val['it'] += 1
                    print("Iteration "+str(val['it'])+":\tMax Residual: "+str(np.max(np.abs(res(val['xi'])))))
                    return val
            else:
                def body(val):
                    val['dxi'] = LS(val['xi'],*val['args'])
                    val['xi'] -= val['dxi']
                    val['it'] += 1
                    return val

        self._nlls = jit(lambda val: lax.while_loop(cond,body,val))
        self._compiled = False

    def run(self,xiInit,*args):

        if self._dictFlag:
            dxi = np.ones_like(xiInit.toArray())
        else:
            dxi = np.ones_like(xiInit)

        if self.timer:
            import time
            timer = getattr(time,self.timerType)

            if not self._compiled:
                val = {'xi':xiInit,'dxi':dxi,'it':self._maxIter-1,'args':args}
                self._nlls(val)['dxi'].block_until_ready()
                self._compiled = True

            val = {'xi':xiInit,'dxi':dxi,'it':0,'args':args}

            start = timer()
            val = self._nlls(val)
            val['dxi'].block_until_ready()
            stop = timer()

            return val['xi'],val['it'],stop-start

        else:
            val = {'xi':xiInit,'dxi':dxi,'it':0,'args':args}
            val = self._nlls(val)

            self._compiled = True

            return val['xi'],val['it']

class ComponentConstraintGraph:

    def __init__(self,N,E):

        # Check that all edges are connected to valid nodes
        self.nNodes = len(N)
        self.nEdges = len(E)
        for k in range(self.nEdges):
            if not (E[k]['node0'] in N and E[k]['node1'] in N):
                TFCPrint.Error("Error either "+E[k]['node0']+" or "+E[k]['node1']+" is not a valid node. Make sure they appear in the nodes list.")

        # Create all possible source/target pairs. This tells whether node0 is the target or source, node1 will be the opposite.
        import itertools 
        self.targets = list(itertools.product([0, 1], repeat=self.nEdges))

        # Find all targets that are valid trees
        self.goodTargets = []
        for j in range(len(self.targets)):
            flag = True
            adj = onp.zeros((self.nNodes,self.nNodes),dtype=np.int32)
            for k in range(self.nNodes):
                kNode = N[k]
                sources = []
                targets = []
                for g in range(self.nEdges):
                    if E[g]['node0'] == kNode:
                        if self.targets[j][g]:
                            adj[N.index(E[g]['node1']),N.index(E[g]['node0'])] = 1
                    elif E[g]['node1'] == kNode:
                        if not self.targets[j][g]:
                            adj[N.index(E[g]['node0']),N.index(E[g]['node1'])] = 1
            if np.all(np.linalg.matrix_power(adj,self.nNodes) == 0):
                self.goodTargets.append(j)

        # Save nodes and edges for use later
        self.N = N
        self.E = E

    def SaveTrees(self,outputDir,allTrees=False,savePDFs=False):
        import os
        from Html import HTML, Dot

        if allTrees:
            targets = self.targets
        else:
            targets = [self.targets[k] for k in self.goodTargets]

        n = len(targets)

        #: Create the main dot file 
        mainDot = Dot(os.path.join(outputDir,'dotFiles','main'),'main')
        mainDot.dot.node_attr.update(shape='box')
        for k in range(n):
            mainDot.dot.node('tree'+str(k),'Tree '+str(k),href=os.path.join('htmlFiles','tree'+str(k)+'.html'))
        mainDot.Render()

        #: Create the main file HTML
        mainHtml = HTML(os.path.join(outputDir,'main.html'))
        with mainHtml.tag('html'):
            with mainHtml.tag('body'):
                with mainHtml.tag('style'):
                    mainHtml.doc.asis(mainHtml.centerClass)
                mainHtml.doc.stag('img',src=os.path.join('dotFiles','main.svg'),usemap='#main',klass='center')
                mainHtml.doc.asis(mainHtml.ReadFile(os.path.join(outputDir,'dotFiles','main.cmapx')))
        mainHtml.WriteFile()

        #: Create the tree dot files
        for k in range(n):
            treeDot = Dot(os.path.join(outputDir,'dotFiles','tree'+str(k)),'tree'+str(k))
            treeDot.dot.attr(bgcolor='transparent')
            treeDot.dot.node_attr.update(shape='box')
            for j in range(self.nNodes):
                treeDot.dot.node(self.N[j],self.N[j])
            for j in range(self.nEdges):
                if not targets[k][j]:
                    treeDot.dot.edge(self.E[j]['node0'],self.E[j]['node1'],label=self.E[j]['name'])
                else:
                    treeDot.dot.edge(self.E[j]['node1'],self.E[j]['node0'],label=self.E[j]['name'])

            if savePDFs:
                treeDot.Render(formats=['cmapx','svg','pdf'])
            else:
                treeDot.Render()

        #: Create the tree HTML files
        for k in range(n):
            treeHtml = HTML(os.path.join(outputDir,'htmlFiles','tree'+str(k)+'.html'))
            with treeHtml.tag('html'):
                with treeHtml.tag('body'):
                    with treeHtml.tag('style'):
                        treeHtml.doc.asis(treeHtml.centerClass)
                    treeHtml.doc.stag('img',src=os.path.join('..','dotFiles','tree'+str(k)+'.svg'),usemap='#tree'+str(k),klass='center')
                    treeHtml.doc.asis(treeHtml.ReadFile(os.path.join(outputDir,'dotFiles','tree'+str(k)+'.cmapx')))
            treeHtml.WriteFile()


def LS(A,B):
    """ This function performs least-squares using the scaled QR method. """
    S = 1./np.sqrt(np.sum(A*A,0))
    S = np.reshape(S,(A.shape[1],))
    q,r = np.linalg.qr(A.dot(np.diag(S)))
    x = S*np.linalg.multi_dot([_MatPinv(r),q.T,B])
    cn = np.linalg.cond(r)
    return x,cn

def _MatPinv(A):
    """ This function is used to better replicate MATLAB's pseudo-inverse. """
    rcond = onp.max(A.shape)*onp.spacing(np.linalg.norm(A,ord=2))
    return np.linalg.pinv(A,rcond=rcond)

def step(x):
    """ This is the unit step function, but the deriative is defined and equal to 0 at every point. """
    return np.heaviside(x,0)
