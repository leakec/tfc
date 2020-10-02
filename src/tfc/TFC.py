import sys

from jax.config import config
config.update('jax_enable_x64', True)

import numpy as onp
import jax.numpy as np
from jax import core, abstract_arrays, jvp
from jax.interpreters import ad, batching, xla
from jax.lib import xla_client

from .utils import TFCPrint

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

# Custom name generator
def NameGen():
    """ Generates a custom name for the TFC class. """
    if not(hasattr(NameGen,'persist')):
        NameGen.persist = 0
    NameGen.persist += 1
    return "TFC"+str(NameGen.persist)


##
#This is the univariate TFC class. It acts as a container that holds:
#  - The linear map from the domain of the DE to the domain of the free-function.
#  - The necessary autograd code that enables automatic differentiation of the constrained experssion and Jacobians of the residual with respect to the unknown coefficients in the linear combination of basis functions that make up the free function.
#  - Other useful TFC related functions such as collocation point creation.
#In addition, this class ties these methods together to form a utility that enables a higher level of code abstraction
#such that the end-user scripts are simple, clear, and elegant implementations of TFC.
#
# For more information on TFC constrained expressions see:
# - <a href = https://doi.org/10.3390/math8081303><i>The Multivariate Theory of Functional Connections: Theory, Proofs, and Application in Partial Differential Equationshe Pyramid Star Identification Techinque</i></a> 
# \rst
# `The Multivariate Theory of Functional Connections: Theory, Proofs, and Application in Partial Differential Equations <https://doi.org/10.3390/math8081303>`_
# \endrst
class TFC:

    ##
    #This function is the constructor for the univariate TFC class. Its inputs are as follows:
    #    * N - Number of points to use when discretizing the domain.
    #    * nC - Number of functions to remove from the beginning of free function linear expansion. This variable is used to account for basis functions that are linearly dependent on support functions used in the construction of the constrained expressions. It can be expressed in 1 of 2 ways. 
    #           -# As an integer. When expressed as an integer, the first nC basis functions are removed from the free function.
    #           -# As a list or tuple. When expressed as a list or tuple, the basis functions corresponding to the numbers given by the list or tuple are removed from the free function. 
    #    * deg - Degree of the basis function expansion. This number is one less than the number of basis functions used.
    #    * basis - This optional string argument specifies the basis functions to be used. The default is Chebyshev orthogonal polynomails.
    #    * c - This argument acts as the constant in the linear map that maps the DE domain to the basis function domain.
    #    * x0 - This optional argument specifies the beginning of the DE domain. The default value "None" will result in a DE domain that begins at 0.
    #    * z - This optional argument is used to specify the basis function domain discretization. The default value will result in the typical collocation discretiztaion. 
    def __init__(self,N,nC,deg,basis='CP',c=0.,x0=None,z=None):

        # Generate a custom name
        self.name = NameGen()

        # Initialize TFCPrint
        TFCPrint()

        # Store givens
        self.N = N
        self.deg = deg

        if isinstance(nC,np.ndarray):
            self.nC = nC
        elif isinstance(nC,list):
            self.nC = np.array(nC)
        else:
            self.nC = np.arange(nC)
        if self.nC.shape[0] > self.deg:
            TFCPrint.Error("Number of basis functions is less than number of constraints!")

        self.basis = basis

        if x0 is None:
            self.x0 = 0.
        else:
            self.x0 = x0

        if c==0:
            TFCPrint.Error("The value of c you have entered is invalid. Please enter a valid value for c.")
            sys.exit()
        else:
            self.c = c

        # Calculate z points and corresponding x
        if z is None:
            if self.basis in ['CP','LeP']:
                n = self.N-1
                I = np.linspace(0,n,n+1)
                self.z = np.cos(np.pi*(n-I)/float(n))
                self.x = (self.z+1.)/self.c+self.x0
            elif self.basis == 'FS':
                self.z = np.linspace(-np.pi,np.pi,num=self.N)
                self.x = (self.z+np.pi)/self.c+self.x0
            else:
                self.z = np.linspace(0.,1.,num=self.N)
                self.x = self.z/self.c+self.x0
        else:
            if not z.shape[0] == self.N:
                TFCPrint.Error("Input vector z is not the correct size.")
            self.z = z.flatten()
            if self.basis in ['CP','LeP']:
                self.x = (self.z+1.)/self.c+self.x0
            elif self.basis == 'FS':
                self.x = (self.z+np.pi)/self.c+self.x0
            else:
                self.x = self.z/self.c+self.x0

        # Setup the basis function
        if self.basis == 'CP':
            from BF import CP
            self.basisClass = CP(self.z,self.nC,self.deg+1,self.c)
        elif self.basis == 'LeP':
            from BF import LeP
            self.basisClass = LeP(self.z,self.nC,self.deg+1,self.c)
        elif self.basis == 'FS':
            from BF import FS
            self.basisClass = FS(self.z,self.nC,self.deg+1,self.c)
        elif self.basis == 'ELMSigmoid':
            from BF import ELMSigmoid
            self.basisClass = ELMSigmoid(self.z,self.nC,self.deg+1,self.c)
        elif self.basis == 'ELMTanh':
            from BF import ELMTanh
            self.basisClass = ELMTanh(self.z,self.nC,self.deg+1,self.c)
        elif self.basis == 'ELMSin':
            from BF import ELMSin
            self.basisClass = ELMSin(self.z,self.nC,self.deg+1,self.c)
        elif self.basis == 'ELMSwish':
            from BF import ELMSwish
            self.basisClass = ELMSwish(self.z,self.nC,self.deg+1,self.c)
        else:
            TFCPrint.Error("Invalid basis selection. Please select a valid basis")

        self.SetupJax()

    ## This function returns the a JAX function that returns a matrix of the basis functions evaluated at each discretization point.
    #  This function can be automatically differentiated via JAX commands. The returned function pointer has the following arguments:
    #     * x - The discretization points. Note that if useVal=False, then this argument is not used. The TFC.z points are used instead.
    #     * full - This optional boolean argument when set to True will ignore the basis functions removed by the nC argument in the TFC constructor. The default is False.
    #     * useVal - This optional boolean argument when set to True will use the values of x passed into the function. Otherwise, it will use the TFC.z points. Note that this behavior is desired as many times we want to compute dH/dx even though H is really a function of z. Using useVal=False (the default) this desired behavior is achieved. 
    def H(self,x,full=False,useVal=False):
        return self.Hjax(x,full=full,useVal=useVal)
    def dH(self,x,full=False,useVal=False):
        """ This function returns a pointer to the deriative of H. See documentation of H for more details. """
        return self.dHjax(x,full=full,useVal=useVal)
    def d2H(self,x,full=False,useVal=False):
        """ This function returns a pointer to the second deriative of H. See documentation of H for more details. """
        return self.d2Hjax(x,full=full,useVal=useVal)
    def d3H(self,x,full=False,useVal=False):
        """ This function returns a pointer to the third deriative of H. See documentation of H for more details. """
        return self.d3Hjax(x,full=full,useVal=useVal)
    def d4H(self,x,full=False,useVal=False):
        """ This function returns a pointer to the fourth deriative of H. See documentation of H for more details. """
        return self.d4Hjax(x,full=full,useVal=useVal)
    def d8H(self,x,full=False,useVal=False):
        """ This function returns a pointer to the eighth deriative of H. See documentation of H for more details. """
        return self.d8Hjax(x,full=full,useVal=useVal)

    def RepMat(self,varIn,dim=1):
        """ This function is used to replicate a vector along the dimension specified by dim to create a matrix
            the same size as the H matrix."""
        if dim == 1:
            if not isinstance(self.nC,tuple):
                return np.tile(varIn,(1,self.deg+1-self.nC))
            else:
                return np.tile(varIn,(1,self.deg+1-self.nC.__len__()))
        elif dim == 0:
            return np.tile(varIn,(self.N,1))
        else:
            TFCPrint.Error('Invalid dimension')

    def LS(self,A,B):
        """ This function performs least-squares using the scaled QR method. """
        S = 1./np.sqrt(np.sum(A*A,0))
        S = np.reshape(S,(A.shape[1],))
        q,r = np.linalg.qr(A.dot(np.diag(S)))
        x = S*np.linalg.multi_dot([self.MatPinv(r),q.T,B])
        cn = np.linalg.cond(r)
        return x,cn

    def MatPinv(self,A):
        """ This function is used to better replicate MATLAB's pseudo-inverse. """
        rcond = onp.max(A.shape)*onp.spacing(np.linalg.norm(A,ord=2))
        return np.linalg.pinv(A,rcond=rcond)

    def step(self,x):
        """ This is the unit step function, but the deriative is defined and equal to 0 at every point. """
        return np.heaviside(x,0)

    @staticmethod
    def egrad(g,j=0):
        """ This function mimics egrad from autograd. """
        def wrapped(*args):
            tans = tuple([onp.ones(args[i].shape) if i == j else onp.zeros(args[i].shape) for i in range(len(args)) ])
            _,x_bar = jvp(g,args,tans)
            return x_bar
        return wrapped

    def SetupJax(self):
        """ This function is used internally by TFC to setup JAX primatives and create desired behavior when taking derivatives of TFC constrained expressions. """

        # Helper functions
        def _constant_bool(c, a):
          return xla_client.ops.Constant(c, bool(a))
        def _constant_s32_scalar(c, a):
          return xla_client.ops.Constant(c, int(a))
        def _unpack_builder(c):
          # If `c` is a ComputationBuilder object, extracts the underlying XlaBuilder.
          return getattr(c, "_builder", c)

        # Regiser XLA function
        obj = self.basisClass.xlaCapsule
        xlaName = "BasisFunc"+str(self.basisClass.identifier)
        xlaName = xlaName.encode('utf-8')
        xla_client.register_custom_call_target(xlaName,obj,platform='cpu')

        # Create primitives
        H_p = core.Primitive("H")
        dH_p = core.Primitive("dH")
        d2H_p = core.Primitive("d2H")
        d3H_p = core.Primitive("d3H")
        d4H_p = core.Primitive("d4H")
        d5H_p = core.Primitive("d5H")
        d6H_p = core.Primitive("d6H")
        d7H_p = core.Primitive("d7H")
        d8H_p = core.Primitive("d8H")

        def Hjax(x,full=False,useVal=False):
            return H_p.bind(x,full=full,useVal=useVal)
        def dHjax(x,full=False,useVal=False):
            return dH_p.bind(x,full=full,useVal=useVal)
        def d2Hjax(x,full=False,useVal=False):
            return d2H_p.bind(x,full=full,useVal=useVal)
        def d3Hjax(x,full=False,useVal=False):
            return d3H_p.bind(x,full=full,useVal=useVal)
        def d4Hjax(x,full=False,useVal=False):
            return d4H_p.bind(x,full=full,useVal=useVal)
        def d5Hjax(x,full=False,useVal=False):
            return d5H_p.bind(x,full=full,useVal=useVal)
        def d6Hjax(x,full=False,useVal=False):
            return d6H_p.bind(x,full=full,useVal=useVal)
        def d7Hjax(x,full=False,useVal=False):
            return d7H_p.bind(x,full=full,useVal=useVal)
        def d8Hjax(x,full=False,useVal=False):
            return d8H_p.bind(x,full=full,useVal=useVal)

        # Implicit translation
        def H_impl(x,full=False,useVal=False):
            return self.basisClass.H(x.flatten(),0,full,useVal)
        def dH_impl(x,full=False,useVal=False):
            return self.basisClass.H(x.flatten(),1,full,useVal)
        def d2H_impl(x,full=False,useVal=False):
            return self.basisClass.H(x.flatten(),2,full,useVal)
        def d3H_impl(x,full=False,useVal=False):
            return self.basisClass.H(x.flatten(),3,full,useVal)
        def d4H_impl(x,full=False,useVal=False):
            return self.basisClass.H(x.flatten(),4,full,useVal)
        def d5H_impl(x,full=False,useVal=False):
            return self.basisClass.H(x.flatten(),5,full,useVal)
        def d6H_impl(x,full=False,useVal=False):
            return self.basisClass.H(x.flatten(),6,full,useVal)
        def d7H_impl(x,full=False,useVal=False):
            return self.basisClass.H(x.flatten(),7,full,useVal)
        def d8H_impl(x,full=False,useVal=False):
            return self.basisClass.H(x.flatten(),8,full,useVal)
        
        H_p.def_impl(H_impl)
        dH_p.def_impl(dH_impl)
        d2H_p.def_impl(d2H_impl)
        d3H_p.def_impl(d3H_impl)
        d4H_p.def_impl(d4H_impl)
        d5H_p.def_impl(d5H_impl)
        d6H_p.def_impl(d6H_impl)
        d7H_p.def_impl(d7H_impl)
        d8H_p.def_impl(d8H_impl)

        # Abstract evaluation
        def H_abstract_eval(x,full=False,useVal=False):
            if useVal:
                dim0 = x.shape[0]
            else: 
                dim0 = self.basisClass.n
            if full:
                dim1 = self.basisClass.m
            else:
                dim1 = self.basisClass.m-self.basisClass.numC
            return abstract_arrays.ShapedArray((dim0,dim1),x.dtype)

        H_p.def_abstract_eval(H_abstract_eval)
        dH_p.def_abstract_eval(H_abstract_eval)
        d2H_p.def_abstract_eval(H_abstract_eval)
        d3H_p.def_abstract_eval(H_abstract_eval)
        d4H_p.def_abstract_eval(H_abstract_eval)
        d5H_p.def_abstract_eval(H_abstract_eval)
        d6H_p.def_abstract_eval(H_abstract_eval)
        d7H_p.def_abstract_eval(H_abstract_eval)
        d8H_p.def_abstract_eval(H_abstract_eval)

        # XLA compilation
        def H_xla(c,x,full=False,useVal=False):
            c = _unpack_builder(c)
            x_shape = c.get_shape(x)
            dims = x_shape.dimensions()
            dtype = x_shape.element_type()
            if useVal:
                dim0 = dims[0]
            else: 
                dim0 = self.basisClass.n
            if full:
                dim1 = self.basisClass.m
            else:
                dim1 = self.basisClass.m-self.basisClass.numC
            return xla_client.ops.CustomCall(c, xlaName, (_constant_s32_scalar(c,self.basisClass.identifier),
                                                            x,
                                                            _constant_s32_scalar(c,0),
                                                            _constant_bool(c,full),
                                                            _constant_bool(c,useVal),
                                                            _constant_s32_scalar(c,dim0),
                                                            _constant_s32_scalar(c,dim1)
                                                         ),
                                                         xla_client.Shape.array_shape(dtype,(dim0,dim1)))
        def dH_xla(c,x,full=False,useVal=False):
            c = _unpack_builder(c)
            x_shape = c.get_shape(x)
            dims = x_shape.dimensions()
            dtype = x_shape.element_type()
            if useVal:
                dim0 = dims[0]
            else: 
                dim0 = self.basisClass.n
            if full:
                dim1 = self.basisClass.m
            else:
                dim1 = self.basisClass.m-self.basisClass.numC
            return xla_client.ops.CustomCall(c, xlaName, (_constant_s32_scalar(c,self.basisClass.identifier),
                                                            x,
                                                            _constant_s32_scalar(c,1),
                                                            _constant_bool(c,full),
                                                            _constant_bool(c,useVal),
                                                            _constant_s32_scalar(c,dim0),
                                                            _constant_s32_scalar(c,dim1)
                                                         ),
                                                         xla_client.Shape.array_shape(dtype,(dim0,dim1)))
        def d2H_xla(c,x,full=False,useVal=False):
            c = _unpack_builder(c)
            x_shape = c.get_shape(x)
            dims = x_shape.dimensions()
            dtype = x_shape.element_type()
            if useVal:
                dim0 = dims[0]
            else: 
                dim0 = self.basisClass.n
            if full:
                dim1 = self.basisClass.m
            else:
                dim1 = self.basisClass.m-self.basisClass.numC
            return xla_client.ops.CustomCall(c, xlaName, (_constant_s32_scalar(c,self.basisClass.identifier),
                                                            x,
                                                            _constant_s32_scalar(c,2),
                                                            _constant_bool(c,full),
                                                            _constant_bool(c,useVal),
                                                            _constant_s32_scalar(c,dim0),
                                                            _constant_s32_scalar(c,dim1)
                                                         ),
                                                         xla_client.Shape.array_shape(dtype,(dim0,dim1)))
        def d3H_xla(c,x,full=False,useVal=False):
            c = _unpack_builder(c)
            x_shape = c.get_shape(x)
            dims = x_shape.dimensions()
            dtype = x_shape.element_type()
            if useVal:
                dim0 = dims[0]
            else: 
                dim0 = self.basisClass.n
            if full:
                dim1 = self.basisClass.m
            else:
                dim1 = self.basisClass.m-self.basisClass.numC
            return xla_client.ops.CustomCall(c, xlaName, (_constant_s32_scalar(c,self.basisClass.identifier),
                                                            x,
                                                            _constant_s32_scalar(c,3),
                                                            _constant_bool(c,full),
                                                            _constant_bool(c,useVal),
                                                            _constant_s32_scalar(c,dim0),
                                                            _constant_s32_scalar(c,dim1)
                                                         ),
                                                         xla_client.Shape.array_shape(dtype,(dim0,dim1)))
        def d4H_xla(c,x,full=False,useVal=False):
            c = _unpack_builder(c)
            x_shape = c.get_shape(x)
            dims = x_shape.dimensions()
            dtype = x_shape.element_type()
            if useVal:
                dim0 = dims[0]
            else: 
                dim0 = self.basisClass.n
            if full:
                dim1 = self.basisClass.m
            else:
                dim1 = self.basisClass.m-self.basisClass.numC
            return xla_client.ops.CustomCall(c, xlaName, (_constant_s32_scalar(c,self.basisClass.identifier),
                                                            x,
                                                            _constant_s32_scalar(c,4),
                                                            _constant_bool(c,full),
                                                            _constant_bool(c,useVal),
                                                            _constant_s32_scalar(c,dim0),
                                                            _constant_s32_scalar(c,dim1)
                                                         ),
                                                         xla_client.Shape.array_shape(dtype,(dim0,dim1)))
        def d5H_xla(c,x,full=False,useVal=False):
            c = _unpack_builder(c)
            x_shape = c.get_shape(x)
            dims = x_shape.dimensions()
            dtype = x_shape.element_type()
            if useVal:
                dim0 = dims[0]
            else: 
                dim0 = self.basisClass.n
            if full:
                dim1 = self.basisClass.m
            else:
                dim1 = self.basisClass.m-self.basisClass.numC
            return xla_client.ops.CustomCall(c, xlaName, (_constant_s32_scalar(c,self.basisClass.identifier),
                                                            x,
                                                            _constant_s32_scalar(c,5),
                                                            _constant_bool(c,full),
                                                            _constant_bool(c,useVal),
                                                            _constant_s32_scalar(c,dim0),
                                                            _constant_s32_scalar(c,dim1)
                                                         ),
                                                         xla_client.Shape.array_shape(dtype,(dim0,dim1)))
        def d6H_xla(c,x,full=False,useVal=False):
            c = _unpack_builder(c)
            x_shape = c.get_shape(x)
            dims = x_shape.dimensions()
            dtype = x_shape.element_type()
            if useVal:
                dim0 = dims[0]
            else: 
                dim0 = self.basisClass.n
            if full:
                dim1 = self.basisClass.m
            else:
                dim1 = self.basisClass.m-self.basisClass.numC
            return xla_client.ops.CustomCall(c, xlaName, (_constant_s32_scalar(c,self.basisClass.identifier),
                                                            x,
                                                            _constant_s32_scalar(c,6),
                                                            _constant_bool(c,full),
                                                            _constant_bool(c,useVal),
                                                            _constant_s32_scalar(c,dim0),
                                                            _constant_s32_scalar(c,dim1)
                                                         ),
                                                         xla_client.Shape.array_shape(dtype,(dim0,dim1)))
        def d7H_xla(c,x,full=False,useVal=False):
            c = _unpack_builder(c)
            x_shape = c.get_shape(x)
            dims = x_shape.dimensions()
            dtype = x_shape.element_type()
            if useVal:
                dim0 = dims[0]
            else: 
                dim0 = self.basisClass.n
            if full:
                dim1 = self.basisClass.m
            else:
                dim1 = self.basisClass.m-self.basisClass.numC
            return xla_client.ops.CustomCall(c, xlaName, (_constant_s32_scalar(c,self.basisClass.identifier),
                                                            x,
                                                            _constant_s32_scalar(c,7),
                                                            _constant_bool(c,full),
                                                            _constant_bool(c,useVal),
                                                            _constant_s32_scalar(c,dim0),
                                                            _constant_s32_scalar(c,dim1)
                                                         ),
                                                         xla_client.Shape.array_shape(dtype,(dim0,dim1)))
        def d8H_xla(c,x,full=False,useVal=False):
            c = _unpack_builder(c)
            x_shape = c.get_shape(x)
            dims = x_shape.dimensions()
            dtype = x_shape.element_type()
            if useVal:
                dim0 = dims[0]
            else: 
                dim0 = self.basisClass.n
            if full:
                dim1 = self.basisClass.m
            else:
                dim1 = self.basisClass.m-self.basisClass.numC
            return xla_client.ops.CustomCall(c, xlaName, (_constant_s32_scalar(c,self.basisClass.identifier),
                                                            x,
                                                            _constant_s32_scalar(c,8),
                                                            _constant_bool(c,full),
                                                            _constant_bool(c,useVal),
                                                            _constant_s32_scalar(c,dim0),
                                                            _constant_s32_scalar(c,dim1)
                                                         ),
                                                         xla_client.Shape.array_shape(dtype,(dim0,dim1)))

        xla.backend_specific_translations['cpu'][H_p] = H_xla
        xla.backend_specific_translations['cpu'][dH_p] = dH_xla
        xla.backend_specific_translations['cpu'][d2H_p] = d2H_xla
        xla.backend_specific_translations['cpu'][d3H_p] = d3H_xla
        xla.backend_specific_translations['cpu'][d4H_p] = d4H_xla
        xla.backend_specific_translations['cpu'][d5H_p] = d5H_xla
        xla.backend_specific_translations['cpu'][d6H_p] = d6H_xla
        xla.backend_specific_translations['cpu'][d7H_p] = d7H_xla
        xla.backend_specific_translations['cpu'][d8H_p] = d8H_xla

        # Define batching translation
        def H_batch(vec,batch,full=False,useVal=False):
            return Hjax(*vec,full=full,useVal=useVal), batch[0]
        def dH_batch(vec,batch,full=False,useVal=False):
            return dHjax(*vec,full=full,useVal=useVal), batch[0]
        def d2H_batch(vec,batch,full=False,useVal=False):
            return d2Hjax(*vec,full=full,useVal=useVal), batch[0]
        def d3H_batch(vec,batch,full=False,useVal=False):
            return d3Hjax(*vec,full=full,useVal=useVal), batch[0]
        def d4H_batch(vec,batch,full=False,useVal=False):
            return d4Hjax(*vec,full=full,useVal=useVal), batch[0]
        def d5H_batch(vec,batch,full=False,useVal=False):
            return d5Hjax(*vec,full=full,useVal=useVal), batch[0]
        def d6H_batch(vec,batch,full=False,useVal=False):
            return d6Hjax(*vec,full=full,useVal=useVal), batch[0]
        def d7H_batch(vec,batch,full=False,useVal=False):
            return d7Hjax(*vec,full=full,useVal=useVal), batch[0]
        def d8H_batch(vec,batch,full=False,useVal=False):
            return d8Hjax(*vec,full=full,useVal=useVal), batch[0]

        batching.primitive_batchers[H_p] = H_batch
        batching.primitive_batchers[dH_p] = dH_batch
        batching.primitive_batchers[d2H_p] = d2H_batch
        batching.primitive_batchers[d3H_p] = d3H_batch
        batching.primitive_batchers[d4H_p] = d4H_batch
        batching.primitive_batchers[d5H_p] = d5H_batch
        batching.primitive_batchers[d6H_p] = d6H_batch
        batching.primitive_batchers[d7H_p] = d7H_batch
        batching.primitive_batchers[d8H_p] = d8H_batch

        # Define jacobain vector product
        def H_jvp(arg_vals,arg_tans,full=False,useVal=False):
            x = arg_vals[0]
            dx = arg_tans[0]
            if not (dx is ad.Zero):
                if type(dx) is batching.BatchTracer:
                    flag = onp.any(dx.val != 0)
                else:
                    flag = onp.any(dx != 0)
                if flag:
                    if len(dx.shape) == 1:
                        out_tans = dHjax(x.flatten(),full=full,useVal=useVal)*np.expand_dims(dx,1)
                    else:
                        out_tans = dHjax(x.flatten(),full=full,useVal=useVal)*dx
            else:
                if useVal:
                    dim0 = x.shape[0]
                else: 
                    dim0 = self.basisClass.n
                if full:
                    dim1 = self.basisClass.m
                else:
                    dim1 = self.basisClass.m-self.basisClass.numC
                out_tans = np.zeros((dim0,dim1))
            return (Hjax(x.flatten(),full=full,useVal=useVal),out_tans)
        def dH_jvp(arg_vals,arg_tans,full=False,useVal=False):
            x = arg_vals[0]
            dx = arg_tans[0]
            if not (dx is ad.Zero):
                if type(dx) is batching.BatchTracer:
                    flag = onp.any(dx.val != 0)
                else:
                    flag = onp.any(dx != 0)
                if flag:
                    if len(dx.shape) == 1:
                        out_tans = d2Hjax(x.flatten(),full=full,useVal=useVal)*np.expand_dims(dx,1)
                    else:
                        out_tans = d2Hjax(x.flatten(),full=full,useVal=useVal)*dx
            else:
                if useVal:
                    dim0 = x.shape[0]
                else: 
                    dim0 = self.basisClass.n
                if full:
                    dim1 = self.basisClass.m
                else:
                    dim1 = self.basisClass.m-self.basisClass.numC
                out_tans = np.zeros((dim0,dim1))
            return (dHjax(x.flatten(),full=full,useVal=useVal),out_tans)
        def d2H_jvp(arg_vals,arg_tans,full=False,useVal=False):
            x = arg_vals[0]
            dx = arg_tans[0]
            if not (dx is ad.Zero):
                if type(dx) is batching.BatchTracer:
                    flag = onp.any(dx.val != 0)
                else:
                    flag = onp.any(dx != 0)
                if flag:
                    if len(dx.shape) == 1:
                        out_tans = d3Hjax(x.flatten(),full=full,useVal=useVal)*np.expand_dims(dx,1)
                    else:
                        out_tans = d3Hjax(x.flatten(),full=full,useVal=useVal)*dx
            else:
                if useVal:
                    dim0 = x.shape[0]
                else: 
                    dim0 = self.basisClass.n
                if full:
                    dim1 = self.basisClass.m
                else:
                    dim1 = self.basisClass.m-self.basisClass.numC
                out_tans = np.zeros((dim0,dim1))
            return (d2Hjax(x.flatten(),full=full,useVal=useVal),out_tans)
        def d3H_jvp(arg_vals,arg_tans,full=False,useVal=False):
            x = arg_vals[0]
            dx = arg_tans[0]
            if not (dx is ad.Zero):
                if type(dx) is batching.BatchTracer:
                    flag = onp.any(dx.val != 0)
                else:
                    flag = onp.any(dx != 0)
                if flag:
                    if len(dx.shape) == 1:
                        out_tans = d4Hjax(x.flatten(),full=full,useVal=useVal)*np.expand_dims(dx,1)
                    else:
                        out_tans = d4Hjax(x.flatten(),full=full,useVal=useVal)*dx
            else:
                if useVal:
                    dim0 = x.shape[0]
                else: 
                    dim0 = self.basisClass.n
                if full:
                    dim1 = self.basisClass.m
                else:
                    dim1 = self.basisClass.m-self.basisClass.numC
                out_tans = np.zeros((dim0,dim1))
            return (d3Hjax(x.flatten(),full=full,useVal=useVal),out_tans)
        def d4H_jvp(arg_vals,arg_tans,full=False,useVal=False):
            x = arg_vals[0]
            dx = arg_tans[0]
            if not (dx is ad.Zero):
                if type(dx) is batching.BatchTracer:
                    flag = onp.any(dx.val != 0)
                else:
                    flag = onp.any(dx != 0)
                if flag:
                    if len(dx.shape) == 1:
                        out_tans = d5Hjax(x.flatten(),full=full,useVal=useVal)*np.expand_dims(dx,1)
                    else:
                        out_tans = d5Hjax(x.flatten(),full=full,useVal=useVal)*dx
            else:
                if useVal:
                    dim0 = x.shape[0]
                else: 
                    dim0 = self.basisClass.n
                if full:
                    dim1 = self.basisClass.m
                else:
                    dim1 = self.basisClass.m-self.basisClass.numC
                out_tans = np.zeros((dim0,dim1))
            return (d4Hjax(x.flatten(),full=full,useVal=useVal),out_tans)
        def d5H_jvp(arg_vals,arg_tans,full=False,useVal=False):
            x = arg_vals[0]
            dx = arg_tans[0]
            if not (dx is ad.Zero):
                if type(dx) is batching.BatchTracer:
                    flag = onp.any(dx.val != 0)
                else:
                    flag = onp.any(dx != 0)
                if flag:
                    if len(dx.shape) == 1:
                        out_tans = d6Hjax(x.flatten(),full=full,useVal=useVal)*np.expand_dims(dx,1)
                    else:
                        out_tans = d6Hjax(x.flatten(),full=full,useVal=useVal)*dx
            else:
                if useVal:
                    dim0 = x.shape[0]
                else: 
                    dim0 = self.basisClass.n
                if full:
                    dim1 = self.basisClass.m
                else:
                    dim1 = self.basisClass.m-self.basisClass.numC
                out_tans = np.zeros((dim0,dim1))
            return (d5Hjax(x.flatten(),full=full,useVal=useVal),out_tans)
        def d6H_jvp(arg_vals,arg_tans,full=False,useVal=False):
            x = arg_vals[0]
            dx = arg_tans[0]
            if not (dx is ad.Zero):
                if type(dx) is batching.BatchTracer:
                    flag = onp.any(dx.val != 0)
                else:
                    flag = onp.any(dx != 0)
                if flag:
                    if len(dx.shape) == 1:
                        out_tans = d7Hjax(x.flatten(),full=full,useVal=useVal)*np.expand_dims(dx,1)
                    else:
                        out_tans = d7Hjax(x.flatten(),full=full,useVal=useVal)*dx
            else:
                if useVal:
                    dim0 = x.shape[0]
                else: 
                    dim0 = self.basisClass.n
                if full:
                    dim1 = self.basisClass.m
                else:
                    dim1 = self.basisClass.m-self.basisClass.numC
                out_tans = np.zeros((dim0,dim1))
            return (d6Hjax(x.flatten(),full=full,useVal=useVal),out_tans)
        def d7H_jvp(arg_vals,arg_tans,full=False,useVal=False):
            x = arg_vals[0]
            dx = arg_tans[0]
            if not (dx is ad.Zero):
                if type(dx) is batching.BatchTracer:
                    flag = onp.any(dx.val != 0)
                else:
                    flag = onp.any(dx != 0)
                if flag:
                    if len(dx.shape) == 1:
                        out_tans = d8Hjax(x.flatten(),full=full,useVal=useVal)*np.expand_dims(dx,1)
                    else:
                        out_tans = d8Hjax(x.flatten(),full=full,useVal=useVal)*dx
            else:
                if useVal:
                    dim0 = x.shape[0]
                else: 
                    dim0 = self.basisClass.n
                if full:
                    dim1 = self.basisClass.m
                else:
                    dim1 = self.basisClass.m-self.basisClass.numC
                out_tans = np.zeros((dim0,dim1))
            return (d7Hjax(x.flatten(),full=full,useVal=useVal),out_tans)

        ad.primitive_jvps[H_p] = H_jvp
        ad.primitive_jvps[dH_p] = dH_jvp
        ad.primitive_jvps[d2H_p] = d2H_jvp
        ad.primitive_jvps[d3H_p] = d3H_jvp
        ad.primitive_jvps[d4H_p] = d4H_jvp
        ad.primitive_jvps[d5H_p] = d5H_jvp
        ad.primitive_jvps[d6H_p] = d6H_jvp
        ad.primitive_jvps[d7H_p] = d7H_jvp

        # Provide pointers from TFC class
        self.Hjax = Hjax
        self.dHjax = dHjax
        self.d2Hjax = d2Hjax
        self.d3Hjax = d3Hjax
        self.d4Hjax = d4Hjax
        self.d8Hjax = d5Hjax

##
# This class combines TFC classes together so that multiple basis functions can be used 
# simultaneously in the solution. Note, that this class is not yet complete.
# TODO: If the two classes have different z values, then H0 = H(z[0],useVal=True) is not
# what we want.
class HybridTFC:

    def __init__(self,tfcClasses):
        if not all([k.N == tfcClasses[0].N for k in tfcClasses]):
            TFCPrint.Error("Not all TFC classes provided have the same number of points.")
        self._tfcClasses = tfcClasses

    def H(self,x,full=False,useVal=False):
        """ This function returns a pointer to a concatenated matrix of the H matrices for each of the tfcClasses provided at initialization. """
        if useVal and (isinstance(x,tuple) or isinstance(x,list)):
            return np.hstack([k.Hjax(x[j],full=full,useVal=useVal) for j,k in enumerate(self._tfcClasses)])
        else:
            return np.hstack([k.Hjax(x,full=full,useVal=useVal) for k in self._tfcClasses])
    def dH(self,x,full=False,useVal=False):
        """ This function returns a pointer to the deriative of H. See documentation of H for more details. """
        if useVal and (isinstance(x,tuple) or isinstance(x,list)):
            return np.hstack([k.dHjax(x[j],full=full,useVal=useVal) for j,k in enumerate(self._tfcClasses)])
        else:
            return np.hstack([k.dHjax(x,full=full,useVal=useVal) for k in self._tfcClasses])
    def d2H(self,x,full=False,useVal=False):
        """ This function returns a pointer to the second deriative of H. See documentation of H for more details. """
        if useVal and (isinstance(x,tuple) or isinstance(x,list)):
            return np.hstack([k.d2Hjax(x[j],full=full,useVal=useVal) for j,k in enumerate(self._tfcClasses)])
        else:
            return np.hstack([k.d2Hjax(x,full=full,useVal=useVal) for k in self._tfcClasses])
    def d3H(self,x,full=False,useVal=False):
        """ This function returns a pointer to the third deriative of H. See documentation of H for more details. """
        if useVal and (isinstance(x,tuple) or isinstance(x,list)):
            return np.hstack([k.d3Hjax(x[j],full=full,useVal=useVal) for j,k in enumerate(self._tfcClasses)])
        else:
            return np.hstack([k.d3Hjax(x,full=full,useVal=useVal) for k in self._tfcClasses])
    def d4H(self,x,full=False,useVal=False):
        """ This function returns a pointer to the fourth deriative of H. See documentation of H for more details. """
        if useVal and (isinstance(x,tuple) or isinstance(x,list)):
            return np.hstack([k.d4Hjax(x[j],full=full,useVal=useVal) for j,k in enumerate(self._tfcClasses)])
        else:
            return np.hstack([k.d4Hjax(x,full=full,useVal=useVal) for k in self._tfcClasses])
    def d8H(self,x,full=False,useVal=False):
        """ This function returns a pointer to the eighth deriative of H. See documentation of H for more details. """
        if useVal and (isinstance(x,tuple) or isinstance(x,list)):
            return np.hstack([k.d8Hjax(x[j],full=full,useVal=useVal) for j,k in enumerate(self._tfcClasses)])
        else:
            return np.hstack([k.d8Hjax(x,full=full,useVal=useVal) for k in self._tfcClasses])
