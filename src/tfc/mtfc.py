from jax.config import config
config.update('jax_enable_x64', True)

import numpy as onp
import jax.numpy as np
from jax import core, abstract_arrays, jvp
from jax.interpreters import ad, batching, xla
from jax.ops import index_update, index
from jax.lib import xla_client

##
#This is the multivariate TFC class. It acts as a container that holds:
#  - The linear map from the domain of the DE to the domain of the free-function.
#  - The necessary JAX code that enables automatic differentiation of the constrained experssion and Jacobians of the residual with respect to the unknown coefficients in the linear combination of basis functions that make up the free function.
#  - Other useful TFC related functions such as collocation point creation.
#In addition, this class ties these methods together to form a utility that enables a higher level of code abstraction
#such that the end-user scripts are simple, clear, and elegant implementations of TFC.
class mtfc:

    ##
    #This function is the constructor for the multivarite TFC class. Its inputs are as follows:
    #    * N - Number of points to use when discretizing the domain.
    #    * nC - Number of functions to remove from the beginning of free function linear expansion. This variable is used to account for basis functions that are linearly dependent on support functions used in the construction of the constrained expressions. The constraints for each dimension can be expressed in 1 of 2 ways. Note that a value of -1 is used to indicate no constraints exist for a particular dimension.
    #           -# As an integer. When expressed as an integer, the first nC basis functions are removed from the free function.
    #           -# As a set of integers. The basis functions corresponding to the numbers given in the set are removed from the free function. 
    #    * deg - Degree of the basis function expansion. This number is one less than the number of basis functions used.
    #    * basis - This optional string argument specifies the basis functions to be used. The default is Chebyshev orthogonal polynomails.
    #    * c - This argument acts as the constant in the linear map that maps the DE domain to the basis function domain.
    #    * x0 - This optional argument specifies the beginning of the DE domain. The default value "None" will result in a DE domain that begins at 0.
    #    * z - This optional argument is used to specify the basis function domain discretization. The default value will result in the typical collocation discretiztaion. 
    def __init__(self,n,nC,deg,dim=2,basis='CP',x0=None,xf=None):

        # Store givens
        self._elm_classes= ['ELMSigmoid','ELMTanh','ELMSin','ELMSwish']
        self.deg = deg
        self.dim = dim
        
        # Set N based on user input
        if isinstance(n,np.ndarray): 
            if not n.flatten().shape[0] == dim:
                TFCPrint.Error("n has length "+str(n.flatten().shape[0])+", but it should be equal to the number of dimensions, "+str(dim)+".")
            self.n = n.astype(np.int32)
        else:
            if not len(n) == dim:
                TFCPrint.Error("n has length "+str(n)+", but it should be equal to the number of dimensions, "+str(dim)+".")
            self.n = np.array(n,dtype=np.int32)
        self.N = int(np.prod(self.n,dtype=np.int32))

        self.basis = basis

        # Set x0 based on user input
        if x0 is None:
            self.x0 = np.zeros(dim)
        else:
            if isinstance(x0,np.ndarray):
                if not x0.flatten().shape[0] == dim:
                    TFCPrint.Error("x0 has length "+str(x0.flatten().shape[0])+", but it should be equal to the number of dimensions, "+str(dim)+".")
                self.x0 = x0
            else:
                if not len(x0) == dim:
                    TFCPrint.Error("x0 has length "+len(x0)+", but it should be equal to the number of dimensions, "+str(dim)+".")
                self.x0 = np.array(x0).flatten()
                if not self.x0.shape[0] == dim:
                    TFCPrint.Error("x0 has length "+str(x0.flatten().shape[0])+", but it should be equal to the number of dimensions, "+str(dim)+".")

        # Set xf based on user input
        if xf is None:
            self.xf = np.zeros(dim)
        else:
            if isinstance(xf,np.ndarray):
                if not xf.flatten().shape[0] == dim:
                    TFCPrint.Error("xf has length "+str(xf.flatten().shape[0])+", but it should be equal to the number of dimensions, "+str(dim)+".")
                self.xf = xf
            else:
                if not len(xf) == dim:
                    TFCPrint.Error("xf has length "+len(xf)+", but it should be equal to the number of dimensions, "+str(dim)+".")
                self.xf = np.array(xf).flatten()
                if not self.xf.shape[0] == dim:
                    TFCPrint.Error("xf has length "+str(xf.flatten().shape[0])+", but it should be equal to the number of dimensions, "+str(dim)+".")

        # Create nC matrix based on user input
        if basis in self._elm_classes:
            if isinstance(nC,np.ndarray):
                self.nC = nC
            elif isinstance(nC,list):
                self.nC = np.array(nC)
            else:
                self.nC = np.arange(nC)
            if self.nC.shape[0] > self.deg:
                TFCPrint.Error("Number of basis functions is less than number of constraints!")
        else:
            if isinstance(nC,np.ndarray) and len(nC.shape) > 1:
                if not nC.shape[0] == self.dim:
                    TFCPrint.Error("nC has "+str(nC.flatten().shape[0])+" rows, but the row number should be equal to the number of dimensions, "+str(dim)+".")
                self.nC = nC.astype(np.int32)
            else:
                if isinstance(nC,np.ndarray):
                    nC = nC.tolist()
                if not len(nC) == dim:
                    TFCPrint.Error("nC has length "+str(len(nC))+", but it should be equal to the number of dimensions, "+str(dim)+".")
                nCmax = 0
                for k in range(dim):
                    if isinstance(nC[k],np.ndarray):
                        nCk = np.array(nC[k]).flatten()
                    else:
                        nCk = np.array([nC[k]]).flatten()
                    if nCk.shape[0] == 1:
                        maxk = nCk[0]
                    else:
                        maxk = nCk.shape[0]
                    if maxk > nCmax:
                        nCmax = maxk

                if nCmax > self.deg:
                    TFCPrint.Error("Number of basis functions is less than the number of constraints!")

                onC = onp.zeros((dim,nCmax))
                for k in range(dim):
                    if isinstance(nC[k],np.ndarray):
                        nCk = np.array(nC[k]).flatten()
                    else:
                        nCk = onp.array([nC[k]]).flatten()
                    n = nCk.shape[0]
                    if n == 1:
                        nCk = onp.arange(nCk[0])
                        n = nCk.shape[0]
                    if n < nCmax:
                        if n == 0:
                            nCk = -1.*onp.ones(nCmax)
                        else:
                            nCk = np.hstack([nCk,-1*np.ones(nCmax-n)])
                    onC[k,:] = nCk.astype(np.int32)
                self.nC = np.array(onC.tolist(),dtype=np.int32)

        # Setup the basis function
        if self.basis == 'CP':
            from .utils.BF import nCP
            self.basisClass = nCP(self.x0,self.xf,self.nC,self.deg+1)
            z0 = -1.; zf = 1.
        elif self.basis == 'LeP':
            from .utils.BF import nLeP
            self.basisClass = nLeP(self.x0,self.xf,self.nC,self.deg+1)
            z0 = -1.; zf = 1.
        elif self.basis == 'FS':
            from .utils.BF import nFS
            self.basisClass = nFS(self.x0,self.xf,self.nC,self.deg+1)
            z0 = -np.pi; zf = np.pi
        elif self.basis == 'ELMSigmoid':
            from .utils.BF import nELMSigmoid
            self.basisClass = nELMSigmoid(self.x0,self.xf,self.nC,self.deg+1)
            z0 = 0.; zf = 1.
        elif self.basis == 'ELMTanh':
            from .utils.BF import nELMTanh
            self.basisClass = nELMTanh(self.x0,self.xf,self.nC,self.deg+1)
            z0 = 0.; zf = 1.
        elif self.basis == 'ELMSin':
            from .utils.BF import nELMSin
            self.basisClass = nELMSin(self.x0,self.xf,self.nC,self.deg+1)
            z0 = 0.; zf = 1.
        elif self.basis == 'ELMSwish':
            from .utils.BF import nELMSwish
            self.basisClass = nELMSwish(self.x0,self.xf,self.nC,self.deg+1)
            z0 = 0.; zf = 1.
        else:
            TFCPrint.Error("Invalid basis selection. Please select a valid basis")

        if self.basisClass.numBasisFunc > self.N:
            TFCPrint.Warning("Warning, you have more basis functions than points!\nThis may lead to large solution errors!")

        self.c = self.basisClass.c

        # Calculate z points and corresponding x
        self.z = onp.zeros((self.dim,self.N))
        x = tuple([onp.zeros(self.N) for x in range(self.dim)])
        if self.basis in ['CP','LeP']:
            for k in range(self.dim):
                nProd = onp.prod(self.n[k+1:])
                nStack = onp.prod(self.n[0:k])
                n = self.n[k]-1
                I = onp.linspace(0,n,n+1).reshape((n+1,1))
                dark = onp.cos(np.pi*(n-I)/float(n))
                dark = onp.hstack([dark]*nProd).flatten()
                self.z[k,:] = onp.array([dark]*nStack).flatten()
                x[k][:] = (self.z[k,:]-z0)/self.c[k] + self.x0[k]
        else:
            for k in range(self.dim):
                nProd = onp.prod(self.n[k+1:])
                nStack = onp.prod(self.n[0:k])
                dark = onp.linspace(z0,zf,num=self.n[k]).reshape((self.n[k],1))
                dark = onp.hstack([dark]*nProd).flatten()
                self.z[k,:] = onp.array([dark]*nStack).flatten()
                x[k][:] = (self.z[k,:]-z0)/self.c[k] + self.x0[k]

        self.z = np.array(self.z.tolist())
        self.x = tuple([np.array(x[k].tolist()) for k in range(self.dim)])

        self.SetupJAX()

    ## This function returns the a JAX function that returns a matrix of the basis functions evaluated at each discretization point.
    #  This function can be automatically differentiated via JAX commands. The returned function pointer has the following arguments:
    #     * x - The discretization points. 
    #     * full - This optional boolean argument when set to True will ignore the basis functions removed by the nC argument in the TFC constructor. The default is False.
    def H(self,*x,full=False):
        return self._Hjax(*x,full=full)
    def Hx(self,*x,full=False):
        """ This function returns a pointer to the deriative of H with respect to x. See documentation of H for more details. """
        return self._Hxjax(*x,full=full)
    def Hx2(self,*x,full=False):
        """ This function returns a pointer to the second deriative of H with respect to x. See documentation of H for more details. """
        return self._Hx2jax(*x,full=full)
    def Hy2(self,*x,full=False):
        """ This function returns a pointer to the second deriative of H with respect to y. See documentation of H for more details. """
        return self._Hy2jax(*x,full=full)
    def Hx2y(self,*x,full=False):
        """ This function returns a pointer of the mixed derivative d^3H/dx^2dy. See documentation of H for more details. """
        return self._Hx2yjax(*x,full=full)
    def Hy(self,*x,full=False):
        """ This function returns a pointer to the deriative of H with respect to y. See documentation of H for more details. """
        return self._Hyjax(*x,full=full)
    def Hxy(self,*x,full=False):
        """ This function returns a pointer of the mixed derivative d^2H/dxdy. See documentation of H for more details. """
        return self._Hxyjax(*x,full=full)
    def Hz(self,*x,full=False):
        """ This function returns a pointer to the deriative of H with respect to z. See documentation of H for more details. """
        return self._Hzjax(*x,full=full)

    def RepMat(self,varIn):
        """ This function is used to replicate a value self.N times to return a vector the same size as one of the dimensions of the z points. """
        return np.tile(varIn,self.N)

    def SetupJAX(self):
        """ This function is used internally by TFC to setup autograd primatives and create desired behavior when taking derivatives of TFC constrained expressions. """

        # Helper functions
        def _constant_bool(c, a):
          return xla_client.ops.Constant(c, bool(a))
        def _constant_s32_scalar(c, a):
          return xla_client.ops.Constant(c, int(a))
        def _constant_array(c, a):
          return xla_client.ops.Constant(c, a)
        def _unpack_builder(c):
          # If `c` is a ComputationBuilder object, extracts the underlying XlaBuilder.
          return getattr(c, "_builder", c)

        # Regiser XLA function
        obj = self.basisClass.xlaCapsule
        xlaName = "BasisFunc"+str(self.basisClass.identifier)
        xlaName = xlaName.encode('utf-8')
        xla_client.register_custom_call_target(xlaName,obj,platform='cpu')

        # Create Primitives
        H_p = core.Primitive("H")
        Hx_p = core.Primitive("Hx")
        Hy_p = core.Primitive("Hy")
        Hz_p = core.Primitive("Hz")
        Hw_p = core.Primitive("Hw")
        Hxy_p = core.Primitive("Hxy")
        Hxz_p = core.Primitive("Hxz")
        Hxw_p = core.Primitive("Hxw")
        Hyz_p = core.Primitive("Hyz")
        Hyw_p = core.Primitive("Hyw")
        Hzw_p = core.Primitive("Hzw")
        Hx2_p = core.Primitive("Hx2")
        Hy2_p = core.Primitive("Hy2")
        Hz2_p = core.Primitive("Hz2")
        Hw2_p = core.Primitive("Hw2")
        Hy3_p = core.Primitive("Hy3")
        Hxy2_p = core.Primitive("Hxy2")
        Hx2y_p = core.Primitive("Hx2y")
        Hx2z_p = core.Primitive("Hx2z")
        Hx3_p = core.Primitive("Hx3")
        Hz3_p = core.Primitive("Hz3")
        Hy2z_p = core.Primitive("Hy2z")
        Hy4_p = core.Primitive("Hy4")
        Hxy3_p = core.Primitive("Hxy3")
        Hx2y2_p = core.Primitive("Hx2y2")
        Hx3y_p = core.Primitive("Hx3y")
        Hx4_p = core.Primitive("Hx4")
        Hy5_p = core.Primitive("Hy5")
        Hxy4_p = core.Primitive("Hxy4")
        Hx2y3_p = core.Primitive("Hx2y3")
        Hx3y2_p = core.Primitive("Hx3y2")
        Hx4y_p = core.Primitive("Hx4y")
        Hx5_p = core.Primitive("Hx5")
        Hy6_p = core.Primitive("Hy6")
        Hxy5_p = core.Primitive("Hxy5")
        Hx2y4_p = core.Primitive("Hx2y4")
        Hx3y3_p = core.Primitive("Hx3y3")
        Hx4y2_p = core.Primitive("Hx4y2")
        Hx5y_p = core.Primitive("Hx5y")
        Hx6_p = core.Primitive("Hx6")
        Hy7_p = core.Primitive("Hy7")
        Hxy6_p = core.Primitive("Hxy6")
        Hx2y5_p = core.Primitive("Hx2y5")
        Hx3y4_p = core.Primitive("Hx3y4")
        Hx4y3_p = core.Primitive("Hx4y3")
        Hx5y2_p = core.Primitive("Hx5y2")
        Hx6y_p = core.Primitive("Hx6y")
        Hx3y5_p = core.Primitive("Hx3y5")
        Hx5y3_p = core.Primitive("Hx5y3")
        Hx6y2_p = core.Primitive("Hx6y2")
        Hx2y6_p = core.Primitive("Hx2y6")
        Hx4y4_p = core.Primitive("Hx4y4")
        Hx2y7_p = core.Primitive("Hx2y7")
        Hx6y3_p = core.Primitive("Hx6y3")
        Hx5y4_p = core.Primitive("Hx5y4")
        Hx4y5_p = core.Primitive("Hx4y5")

        def Hjax(*x,full=False):
                return H_p.bind(*x,full=full)
        def Hxjax(*x,full=False):
                return Hx_p.bind(*x,full=full)
        def Hyjax(*x,full=False):
                return Hy_p.bind(*x,full=full)
        def Hzjax(*x,full=False):
                return Hz_p.bind(*x,full=full)
        def Hwjax(*x,full=False):
                return Hw_p.bind(*x,full=full)
        def Hxyjax(*x,full=False):
                return Hxy_p.bind(*x,full=full)
        def Hxzjax(*x,full=False):
                return Hxz_p.bind(*x,full=full)
        def Hxwjax(*x,full=False):
                return Hxw_p.bind(*x,full=full)
        def Hyzjax(*x,full=False):
                return Hyz_p.bind(*x,full=full)
        def Hywjax(*x,full=False):
                return Hyw_p.bind(*x,full=full)
        def Hzwjax(*x,full=False):
                return Hzw_p.bind(*x,full=full)
        def Hx2jax(*x,full=False):
                return Hx2_p.bind(*x,full=full)
        def Hy2jax(*x,full=False):
                return Hy2_p.bind(*x,full=full)
        def Hz2jax(*x,full=False):
                return Hz2_p.bind(*x,full=full)
        def Hw2jax(*x,full=False):
                return Hw2_p.bind(*x,full=full)
        def Hy3jax(*x,full=False):
                return Hy3_p.bind(*x,full=full)
        def Hxy2jax(*x,full=False):
                return Hxy2_p.bind(*x,full=full)
        def Hx2yjax(*x,full=False):
                return Hx2y_p.bind(*x,full=full)
        def Hx2zjax(*x,full=False):
                return Hx2z_p.bind(*x,full=full)
        def Hx3jax(*x,full=False):
                return Hx3_p.bind(*x,full=full)
        def Hz3jax(*x,full=False):
                return Hz3_p.bind(*x,full=full)
        def Hy2zjax(*x,full=False):
                return Hy2z_p.bind(*x,full=full)
        def Hy4jax(*x,full=False):
                return Hy4_p.bind(*x,full=full)
        def Hxy3jax(*x,full=False):
                return Hxy3_p.bind(*x,full=full)
        def Hx2y2jax(*x,full=False):
                return Hx2y2_p.bind(*x,full=full)
        def Hx3yjax(*x,full=False):
                return Hx3y_p.bind(*x,full=full)
        def Hx4jax(*x,full=False):
                return Hx4_p.bind(*x,full=full)
        def Hy5jax(*x,full=False):
                return Hy5_p.bind(*x,full=full)
        def Hxy4jax(*x,full=False):
                return Hxy4_p.bind(*x,full=full)
        def Hx2y3jax(*x,full=False):
                return Hx2y3_p.bind(*x,full=full)
        def Hx3y2jax(*x,full=False):
                return Hx3y2_p.bind(*x,full=full)
        def Hx4yjax(*x,full=False):
                return Hx4y_p.bind(*x,full=full)
        def Hx5jax(*x,full=False):
                return Hx5_p.bind(*x,full=full)
        def Hy6jax(*x,full=False):
                return Hy6_p.bind(*x,full=full)
        def Hxy5jax(*x,full=False):
                return Hxy5_p.bind(*x,full=full)
        def Hx2y4jax(*x,full=False):
                return Hx2y4_p.bind(*x,full=full)
        def Hx3y3jax(*x,full=False):
                return Hx3y3_p.bind(*x,full=full)
        def Hx4y2jax(*x,full=False):
                return Hx4y2_p.bind(*x,full=full)
        def Hx5yjax(*x,full=False):
                return Hx5y_p.bind(*x,full=full)
        def Hx6jax(*x,full=False):
                return Hx6_p.bind(*x,full=full)
        def Hy7jax(*x,full=False):
                return Hy7_p.bind(*x,full=full)
        def Hxy6jax(*x,full=False):
                return Hxy6_p.bind(*x,full=full)
        def Hx2y5jax(*x,full=False):
                return Hx2y5_p.bind(*x,full=full)
        def Hx3y4jax(*x,full=False):
                return Hx3y4_p.bind(*x,full=full)
        def Hx4y3jax(*x,full=False):
                return Hx4y3_p.bind(*x,full=full)
        def Hx5y2jax(*x,full=False):
                return Hx5y2_p.bind(*x,full=full)
        def Hx6yjax(*x,full=False):
                return Hx6y_p.bind(*x,full=full)
        def Hx3y5jax(*x,full=False):
                return Hx3y5_p.bind(*x,full=full)
        def Hx5y3jax(*x,full=False):
                return Hx5y3_p.bind(*x,full=full)
        def Hx6y2jax(*x,full=False):
                return Hx6y2_p.bind(*x,full=full)
        def Hx2y6jax(*x,full=False):
                return Hx2y6_p.bind(*x,full=full)
        def Hx4y4jax(*x,full=False):
                return Hx4y4_p.bind(*x,full=full)
        def Hx2y7jax(*x,full=False):
                return Hx2y7_p.bind(*x,full=full)
        def Hx6y3jax(*x,full=False):
                return Hx6y3_p.bind(*x,full=full)
        def Hx5y4jax(*x,full=False):
                return Hx5y4_p.bind(*x,full=full)
        def Hx4y5jax(*x,full=False):
                return Hx4y5_p.bind(*x,full=full)

        # Implicit translations
        def H_impl(*x,full=False):
                return self.basisClass.H(np.array(x),np.array([0],dtype=np.int32),full)
        def Hx_impl(*x,full=False):
                return self.basisClass.H(np.array(x),np.array([1],dtype=np.int32),full)
        def Hy_impl(*x,full=False):
                return self.basisClass.H(np.array(x),np.array([0,1],dtype=np.int32),full)
        def Hz_impl(*x,full=False):
                return self.basisClass.H(np.array(x),np.array([0,0,1],dtype=np.int32),full)
        def Hw_impl(*x,full=False):
                return self.basisClass.H(np.array(x),np.array([0,0,0,1],dtype=np.int32),full)
        def Hxy_impl(*x,full=False):
                return self.basisClass.H(np.array(x),np.array([1,1],dtype=np.int32),full)
        def Hxz_impl(*x,full=False):
                return self.basisClass.H(np.array(x),np.array([1,0,1],dtype=np.int32),full)
        def Hxw_impl(*x,full=False):
                return self.basisClass.H(np.array(x),np.array([1,0,0,1],dtype=np.int32),full)
        def Hyz_impl(*x,full=False):
                return self.basisClass.H(np.array(x),np.array([0,1,1],dtype=np.int32),full)
        def Hyw_impl(*x,full=False):
                return self.basisClass.H(np.array(x),np.array([0,1,0,1],dtype=np.int32),full)
        def Hzw_impl(*x,full=False):
                return self.basisClass.H(np.array(x),np.array([0,0,1,1],dtype=np.int32),full)
        def Hx2_impl(*x,full=False):
                return self.basisClass.H(np.array(x),np.array([2],dtype=np.int32),full)
        def Hy2_impl(*x,full=False):
                return self.basisClass.H(np.array(x),np.array([0,2],dtype=np.int32),full)
        def Hz2_impl(*x,full=False):
                return self.basisClass.H(np.array(x),np.array([0,0,2],dtype=np.int32),full)
        def Hw2_impl(*x,full=False):
                return self.basisClass.H(np.array(x),np.array([0,0,0,2],dtype=np.int32),full)
        def Hy3_impl(*x,full=False):
                return self.basisClass.H(np.array(x),np.array([0,3],dtype=np.int32),full)
        def Hxy2_impl(*x,full=False):
                return self.basisClass.H(np.array(x),np.array([1,2],dtype=np.int32),full)
        def Hx2y_impl(*x,full=False):
                return self.basisClass.H(np.array(x),np.array([2,1],dtype=np.int32),full)
        def Hx2z_impl(*x,full=False):
                return self.basisClass.H(np.array(x),np.array([2,0,1],dtype=np.int32),full)
        def Hx3_impl(*x,full=False):
                return self.basisClass.H(np.array(x),np.array([3],dtype=np.int32),full)
        def Hz3_impl(*x,full=False):
                return self.basisClass.H(np.array(x),np.array([0,0,3],dtype=np.int32),full)
        def Hy2z_impl(*x,full=False):
                return self.basisClass.H(np.array(x),np.array([0,2,1],dtype=np.int32),full)
        def Hy4_impl(*x,full=False):
                return self.basisClass.H(np.array(x),np.array([0,4],dtype=np.int32),full)
        def Hxy3_impl(*x,full=False):
                return self.basisClass.H(np.array(x),np.array([1,3],dtype=np.int32),full)
        def Hx2y2_impl(*x,full=False):
                return self.basisClass.H(np.array(x),np.array([2,2],dtype=np.int32),full)
        def Hx3y_impl(*x,full=False):
                return self.basisClass.H(np.array(x),np.array([3,1],dtype=np.int32),full)
        def Hx4_impl(*x,full=False):
                return self.basisClass.H(np.array(x),np.array([4],dtype=np.int32),full)
        def Hy5_impl(*x,full=False):
                return self.basisClass.H(np.array(x),np.array([0,5],dtype=np.int32),full)
        def Hxy4_impl(*x,full=False):
                return self.basisClass.H(np.array(x),np.array([1,4],dtype=np.int32),full)
        def Hx2y3_impl(*x,full=False):
                return self.basisClass.H(np.array(x),np.array([2,3],dtype=np.int32),full)
        def Hx3y2_impl(*x,full=False):
                return self.basisClass.H(np.array(x),np.array([3,2],dtype=np.int32),full)
        def Hx4y_impl(*x,full=False):
                return self.basisClass.H(np.array(x),np.array([4,1],dtype=np.int32),full)
        def Hx5_impl(*x,full=False):
                return self.basisClass.H(np.array(x),np.array([5],dtype=np.int32),full)
        def Hy6_impl(*x,full=False):
                return self.basisClass.H(np.array(x),np.array([0,6],dtype=np.int32),full)
        def Hxy5_impl(*x,full=False):
                return self.basisClass.H(np.array(x),np.array([1,5],dtype=np.int32),full)
        def Hx2y4_impl(*x,full=False):
                return self.basisClass.H(np.array(x),np.array([2,4],dtype=np.int32),full)
        def Hx3y3_impl(*x,full=False):
                return self.basisClass.H(np.array(x),np.array([3,3],dtype=np.int32),full)
        def Hx4y2_impl(*x,full=False):
                return self.basisClass.H(np.array(x),np.array([4,2],dtype=np.int32),full)
        def Hx5y_impl(*x,full=False):
                return self.basisClass.H(np.array(x),np.array([5,1],dtype=np.int32),full)
        def Hx6_impl(*x,full=False):
                return self.basisClass.H(np.array(x),np.array([6],dtype=np.int32),full)
        def Hy7_impl(*x,full=False):
                return self.basisClass.H(np.array(x),np.array([0,7],dtype=np.int32),full)
        def Hxy6_impl(*x,full=False):
                return self.basisClass.H(np.array(x),np.array([1,6],dtype=np.int32),full)
        def Hx2y5_impl(*x,full=False):
                return self.basisClass.H(np.array(x),np.array([2,5],dtype=np.int32),full)
        def Hx3y4_impl(*x,full=False):
                return self.basisClass.H(np.array(x),np.array([3,4],dtype=np.int32),full)
        def Hx4y3_impl(*x,full=False):
                return self.basisClass.H(np.array(x),np.array([4,3],dtype=np.int32),full)
        def Hx5y2_impl(*x,full=False):
                return self.basisClass.H(np.array(x),np.array([5,2],dtype=np.int32),full)
        def Hx6y_impl(*x,full=False):
                return self.basisClass.H(np.array(x),np.array([6,1],dtype=np.int32),full)
        def Hx3y5_impl(*x,full=False):
                return self.basisClass.H(np.array(x),np.array([3,5],dtype=np.int32),full)
        def Hx5y3_impl(*x,full=False):
                return self.basisClass.H(np.array(x),np.array([5,3],dtype=np.int32),full)
        def Hx6y2_impl(*x,full=False):
                return self.basisClass.H(np.array(x),np.array([6,2],dtype=np.int32),full)
        def Hx2y6_impl(*x,full=False):
                return self.basisClass.H(np.array(x),np.array([2,6],dtype=np.int32),full)
        def Hx4y4_impl(*x,full=False):
                return self.basisClass.H(np.array(x),np.array([4,4],dtype=np.int32),full)
        def Hx2y7_impl(*x,full=False):
                return self.basisClass.H(np.array(x),np.array([2,7],dtype=np.int32),full)
        def Hx6y3_impl(*x,full=False):
                return self.basisClass.H(np.array(x),np.array([6,3],dtype=np.int32),full)
        def Hx5y4_impl(*x,full=False):
                return self.basisClass.H(np.array(x),np.array([5,4],dtype=np.int32),full)
        def Hx4y5_impl(*x,full=False):
                return self.basisClass.H(np.array(x),np.array([4,5],dtype=np.int32),full)

        H_p.def_impl(H_impl)
        Hx_p.def_impl(Hx_impl)
        Hy_p.def_impl(Hy_impl)
        Hz_p.def_impl(Hz_impl)
        Hw_p.def_impl(Hw_impl)
        Hxy_p.def_impl(Hxy_impl)
        Hxz_p.def_impl(Hxz_impl)
        Hxw_p.def_impl(Hxw_impl)
        Hyz_p.def_impl(Hyz_impl)
        Hyw_p.def_impl(Hyw_impl)
        Hzw_p.def_impl(Hzw_impl)
        Hx2_p.def_impl(Hx2_impl)
        Hy2_p.def_impl(Hy2_impl)
        Hz2_p.def_impl(Hz2_impl)
        Hw2_p.def_impl(Hw2_impl)
        Hy3_p.def_impl(Hy3_impl)
        Hxy2_p.def_impl(Hxy2_impl)
        Hx2y_p.def_impl(Hx2y_impl)
        Hx2z_p.def_impl(Hx2z_impl)
        Hx3_p.def_impl(Hx3_impl)
        Hz3_p.def_impl(Hz3_impl)
        Hy2z_p.def_impl(Hy2z_impl)
        Hy4_p.def_impl(Hy4_impl)
        Hxy3_p.def_impl(Hxy3_impl)
        Hx2y2_p.def_impl(Hx2y2_impl)
        Hx3y_p.def_impl(Hx3y_impl)
        Hx4_p.def_impl(Hx4_impl)
        Hy5_p.def_impl(Hy5_impl)
        Hxy4_p.def_impl(Hxy4_impl)
        Hx2y3_p.def_impl(Hx2y3_impl)
        Hx3y2_p.def_impl(Hx3y2_impl)
        Hx4y_p.def_impl(Hx4y_impl)
        Hx5_p.def_impl(Hx5_impl)
        Hy6_p.def_impl(Hy6_impl)
        Hxy5_p.def_impl(Hxy5_impl)
        Hx2y4_p.def_impl(Hx2y4_impl)
        Hx3y3_p.def_impl(Hx3y3_impl)
        Hx4y2_p.def_impl(Hx4y2_impl)
        Hx5y_p.def_impl(Hx5y_impl)
        Hx6_p.def_impl(Hx6_impl)
        Hy7_p.def_impl(Hy7_impl)
        Hxy6_p.def_impl(Hxy6_impl)
        Hx2y5_p.def_impl(Hx2y5_impl)
        Hx3y4_p.def_impl(Hx3y4_impl)
        Hx4y3_p.def_impl(Hx4y3_impl)
        Hx5y2_p.def_impl(Hx5y2_impl)
        Hx6y_p.def_impl(Hx6y_impl)
        Hx3y5_p.def_impl(Hx3y5_impl)
        Hx5y3_p.def_impl(Hx5y3_impl)
        Hx6y2_p.def_impl(Hx6y2_impl)
        Hx2y6_p.def_impl(Hx2y6_impl)
        Hx4y4_p.def_impl(Hx4y4_impl)
        Hx2y7_p.def_impl(Hx2y7_impl)
        Hx6y3_p.def_impl(Hx6y3_impl)
        Hx5y4_p.def_impl(Hx5y4_impl)
        Hx4y5_p.def_impl(Hx4y5_impl)

        def H_abstract_eval(*x,full=False):
                dim0 = x[0].shape[0]
                if full:
                        dim1 = self.basisClass.numBasisFuncFull
                else:
                        dim1 = self.basisClass.numBasisFunc
                return abstract_arrays.ShapedArray((dim0,dim1),x[0].dtype)

        H_p.def_abstract_eval(H_abstract_eval)
        Hx_p.def_abstract_eval(H_abstract_eval)
        Hy_p.def_abstract_eval(H_abstract_eval)
        Hz_p.def_abstract_eval(H_abstract_eval)
        Hw_p.def_abstract_eval(H_abstract_eval)
        Hxy_p.def_abstract_eval(H_abstract_eval)
        Hxz_p.def_abstract_eval(H_abstract_eval)
        Hxw_p.def_abstract_eval(H_abstract_eval)
        Hyz_p.def_abstract_eval(H_abstract_eval)
        Hyw_p.def_abstract_eval(H_abstract_eval)
        Hzw_p.def_abstract_eval(H_abstract_eval)
        Hx2_p.def_abstract_eval(H_abstract_eval)
        Hy2_p.def_abstract_eval(H_abstract_eval)
        Hz2_p.def_abstract_eval(H_abstract_eval)
        Hw2_p.def_abstract_eval(H_abstract_eval)
        Hy3_p.def_abstract_eval(H_abstract_eval)
        Hxy2_p.def_abstract_eval(H_abstract_eval)
        Hx2y_p.def_abstract_eval(H_abstract_eval)
        Hx2z_p.def_abstract_eval(H_abstract_eval)
        Hx3_p.def_abstract_eval(H_abstract_eval)
        Hz3_p.def_abstract_eval(H_abstract_eval)
        Hy2z_p.def_abstract_eval(H_abstract_eval)
        Hy4_p.def_abstract_eval(H_abstract_eval)
        Hxy3_p.def_abstract_eval(H_abstract_eval)
        Hx2y2_p.def_abstract_eval(H_abstract_eval)
        Hx3y_p.def_abstract_eval(H_abstract_eval)
        Hx4_p.def_abstract_eval(H_abstract_eval)
        Hy5_p.def_abstract_eval(H_abstract_eval)
        Hxy4_p.def_abstract_eval(H_abstract_eval)
        Hx2y3_p.def_abstract_eval(H_abstract_eval)
        Hx3y2_p.def_abstract_eval(H_abstract_eval)
        Hx4y_p.def_abstract_eval(H_abstract_eval)
        Hx5_p.def_abstract_eval(H_abstract_eval)
        Hy6_p.def_abstract_eval(H_abstract_eval)
        Hxy5_p.def_abstract_eval(H_abstract_eval)
        Hx2y4_p.def_abstract_eval(H_abstract_eval)
        Hx3y3_p.def_abstract_eval(H_abstract_eval)
        Hx4y2_p.def_abstract_eval(H_abstract_eval)
        Hx5y_p.def_abstract_eval(H_abstract_eval)
        Hx6_p.def_abstract_eval(H_abstract_eval)
        Hy7_p.def_abstract_eval(H_abstract_eval)
        Hxy6_p.def_abstract_eval(H_abstract_eval)
        Hx2y5_p.def_abstract_eval(H_abstract_eval)
        Hx3y4_p.def_abstract_eval(H_abstract_eval)
        Hx4y3_p.def_abstract_eval(H_abstract_eval)
        Hx5y2_p.def_abstract_eval(H_abstract_eval)
        Hx6y_p.def_abstract_eval(H_abstract_eval)
        Hx3y5_p.def_abstract_eval(H_abstract_eval)
        Hx5y3_p.def_abstract_eval(H_abstract_eval)
        Hx6y2_p.def_abstract_eval(H_abstract_eval)
        Hx2y6_p.def_abstract_eval(H_abstract_eval)
        Hx4y4_p.def_abstract_eval(H_abstract_eval)
        Hx2y7_p.def_abstract_eval(H_abstract_eval)
        Hx6y3_p.def_abstract_eval(H_abstract_eval)
        Hx5y4_p.def_abstract_eval(H_abstract_eval)
        Hx4y5_p.def_abstract_eval(H_abstract_eval)

        # XLA compilation
        def H_xla(c,*x,full=False):
                c = _unpack_builder(c)
                x_shape = c.get_shape(x[0])
                dims = x_shape.dimensions()
                dtype = x_shape.element_type()
                dim0 = dims[0]
                if full:
                        dim1 = self.basisClass.numBasisFuncFull
                else:
                        dim1 = self.basisClass.numBasisFunc
                return xla_client.ops.CustomCall(c, xlaName, (_constant_s32_scalar(c,self.basisClass.identifier),
                                                              xla_client.ops.ConcatInDim(c,x,0),
                                                              _constant_array(c,np.array([0],dtype=np.int32)),
                                                              _constant_s32_scalar(c,1),
                                                              _constant_bool(c,full),
                                                              _constant_s32_scalar(c,dim0),
                                                              _constant_s32_scalar(c,dim1)
                                                         ),
                                                         xla_client.Shape.array_shape(dtype,(dim0,dim1)))
        def Hx_xla(c,*x,full=False):
                c = _unpack_builder(c)
                x_shape = c.get_shape(x[0])
                dims = x_shape.dimensions()
                dtype = x_shape.element_type()
                dim0 = dims[0]
                if full:
                        dim1 = self.basisClass.numBasisFuncFull
                else:
                        dim1 = self.basisClass.numBasisFunc
                return xla_client.ops.CustomCall(c, xlaName, (_constant_s32_scalar(c,self.basisClass.identifier),
                                                              xla_client.ops.ConcatInDim(c,x,0),
                                                              _constant_array(c,np.array([1],dtype=np.int32)),
                                                              _constant_s32_scalar(c,1),
                                                              _constant_bool(c,full),
                                                              _constant_s32_scalar(c,dim0),
                                                              _constant_s32_scalar(c,dim1)
                                                         ),
                                                         xla_client.Shape.array_shape(dtype,(dim0,dim1)))
        def Hy_xla(c,*x,full=False):
                c = _unpack_builder(c)
                x_shape = c.get_shape(x[0])
                dims = x_shape.dimensions()
                dtype = x_shape.element_type()
                dim0 = dims[0]
                if full:
                        dim1 = self.basisClass.numBasisFuncFull
                else:
                        dim1 = self.basisClass.numBasisFunc
                return xla_client.ops.CustomCall(c, xlaName, (_constant_s32_scalar(c,self.basisClass.identifier),
                                                              xla_client.ops.ConcatInDim(c,x,0),
                                                              _constant_array(c,np.array([0,1],dtype=np.int32)),
                                                              _constant_s32_scalar(c,2),
                                                              _constant_bool(c,full),
                                                              _constant_s32_scalar(c,dim0),
                                                              _constant_s32_scalar(c,dim1)
                                                         ),
                                                         xla_client.Shape.array_shape(dtype,(dim0,dim1)))
        def Hz_xla(c,*x,full=False):
                c = _unpack_builder(c)
                x_shape = c.get_shape(x[0])
                dims = x_shape.dimensions()
                dtype = x_shape.element_type()
                dim0 = dims[0]
                if full:
                        dim1 = self.basisClass.numBasisFuncFull
                else:
                        dim1 = self.basisClass.numBasisFunc
                return xla_client.ops.CustomCall(c, xlaName, (_constant_s32_scalar(c,self.basisClass.identifier),
                                                              xla_client.ops.ConcatInDim(c,x,0),
                                                              _constant_array(c,np.array([0,0,1],dtype=np.int32)),
                                                              _constant_s32_scalar(c,3),
                                                              _constant_bool(c,full),
                                                              _constant_s32_scalar(c,dim0),
                                                              _constant_s32_scalar(c,dim1)
                                                         ),
                                                         xla_client.Shape.array_shape(dtype,(dim0,dim1)))
        def Hw_xla(c,*x,full=False):
                c = _unpack_builder(c)
                x_shape = c.get_shape(x[0])
                dims = x_shape.dimensions()
                dtype = x_shape.element_type()
                dim0 = dims[0]
                if full:
                        dim1 = self.basisClass.numBasisFuncFull
                else:
                        dim1 = self.basisClass.numBasisFunc
                return xla_client.ops.CustomCall(c, xlaName, (_constant_s32_scalar(c,self.basisClass.identifier),
                                                              xla_client.ops.ConcatInDim(c,x,0),
                                                              _constant_array(c,np.array([0,0,0,1],dtype=np.int32)),
                                                              _constant_s32_scalar(c,4),
                                                              _constant_bool(c,full),
                                                              _constant_s32_scalar(c,dim0),
                                                              _constant_s32_scalar(c,dim1)
                                                         ),
                                                         xla_client.Shape.array_shape(dtype,(dim0,dim1)))
        def Hxy_xla(c,*x,full=False):
                c = _unpack_builder(c)
                x_shape = c.get_shape(x[0])
                dims = x_shape.dimensions()
                dtype = x_shape.element_type()
                dim0 = dims[0]
                if full:
                        dim1 = self.basisClass.numBasisFuncFull
                else:
                        dim1 = self.basisClass.numBasisFunc
                return xla_client.ops.CustomCall(c, xlaName, (_constant_s32_scalar(c,self.basisClass.identifier),
                                                              xla_client.ops.ConcatInDim(c,x,0),
                                                              _constant_array(c,np.array([1,1],dtype=np.int32)),
                                                              _constant_s32_scalar(c,2),
                                                              _constant_bool(c,full),
                                                              _constant_s32_scalar(c,dim0),
                                                              _constant_s32_scalar(c,dim1)
                                                         ),
                                                         xla_client.Shape.array_shape(dtype,(dim0,dim1)))
        def Hxz_xla(c,*x,full=False):
                c = _unpack_builder(c)
                x_shape = c.get_shape(x[0])
                dims = x_shape.dimensions()
                dtype = x_shape.element_type()
                dim0 = dims[0]
                if full:
                        dim1 = self.basisClass.numBasisFuncFull
                else:
                        dim1 = self.basisClass.numBasisFunc
                return xla_client.ops.CustomCall(c, xlaName, (_constant_s32_scalar(c,self.basisClass.identifier),
                                                              xla_client.ops.ConcatInDim(c,x,0),
                                                              _constant_array(c,np.array([1,0,1],dtype=np.int32)),
                                                              _constant_s32_scalar(c,3),
                                                              _constant_bool(c,full),
                                                              _constant_s32_scalar(c,dim0),
                                                              _constant_s32_scalar(c,dim1)
                                                         ),
                                                         xla_client.Shape.array_shape(dtype,(dim0,dim1)))
        def Hxw_xla(c,*x,full=False):
                c = _unpack_builder(c)
                x_shape = c.get_shape(x[0])
                dims = x_shape.dimensions()
                dtype = x_shape.element_type()
                dim0 = dims[0]
                if full:
                        dim1 = self.basisClass.numBasisFuncFull
                else:
                        dim1 = self.basisClass.numBasisFunc
                return xla_client.ops.CustomCall(c, xlaName, (_constant_s32_scalar(c,self.basisClass.identifier),
                                                              xla_client.ops.ConcatInDim(c,x,0),
                                                              _constant_array(c,np.array([1,0,0,1],dtype=np.int32)),
                                                              _constant_s32_scalar(c,4),
                                                              _constant_bool(c,full),
                                                              _constant_s32_scalar(c,dim0),
                                                              _constant_s32_scalar(c,dim1)
                                                         ),
                                                         xla_client.Shape.array_shape(dtype,(dim0,dim1)))
        def Hyz_xla(c,*x,full=False):
                c = _unpack_builder(c)
                x_shape = c.get_shape(x[0])
                dims = x_shape.dimensions()
                dtype = x_shape.element_type()
                dim0 = dims[0]
                if full:
                        dim1 = self.basisClass.numBasisFuncFull
                else:
                        dim1 = self.basisClass.numBasisFunc
                return xla_client.ops.CustomCall(c, xlaName, (_constant_s32_scalar(c,self.basisClass.identifier),
                                                              xla_client.ops.ConcatInDim(c,x,0),
                                                              _constant_array(c,np.array([0,1,1],dtype=np.int32)),
                                                              _constant_s32_scalar(c,3),
                                                              _constant_bool(c,full),
                                                              _constant_s32_scalar(c,dim0),
                                                              _constant_s32_scalar(c,dim1)
                                                         ),
                                                         xla_client.Shape.array_shape(dtype,(dim0,dim1)))
        def Hyw_xla(c,*x,full=False):
                c = _unpack_builder(c)
                x_shape = c.get_shape(x[0])
                dims = x_shape.dimensions()
                dtype = x_shape.element_type()
                dim0 = dims[0]
                if full:
                        dim1 = self.basisClass.numBasisFuncFull
                else:
                        dim1 = self.basisClass.numBasisFunc
                return xla_client.ops.CustomCall(c, xlaName, (_constant_s32_scalar(c,self.basisClass.identifier),
                                                              xla_client.ops.ConcatInDim(c,x,0),
                                                              _constant_array(c,np.array([0,1,0,1],dtype=np.int32)),
                                                              _constant_s32_scalar(c,4),
                                                              _constant_bool(c,full),
                                                              _constant_s32_scalar(c,dim0),
                                                              _constant_s32_scalar(c,dim1)
                                                         ),
                                                         xla_client.Shape.array_shape(dtype,(dim0,dim1)))
        def Hzw_xla(c,*x,full=False):
                c = _unpack_builder(c)
                x_shape = c.get_shape(x[0])
                dims = x_shape.dimensions()
                dtype = x_shape.element_type()
                dim0 = dims[0]
                if full:
                        dim1 = self.basisClass.numBasisFuncFull
                else:
                        dim1 = self.basisClass.numBasisFunc
                return xla_client.ops.CustomCall(c, xlaName, (_constant_s32_scalar(c,self.basisClass.identifier),
                                                              xla_client.ops.ConcatInDim(c,x,0),
                                                              _constant_array(c,np.array([0,0,1,1],dtype=np.int32)),
                                                              _constant_s32_scalar(c,4),
                                                              _constant_bool(c,full),
                                                              _constant_s32_scalar(c,dim0),
                                                              _constant_s32_scalar(c,dim1)
                                                         ),
                                                         xla_client.Shape.array_shape(dtype,(dim0,dim1)))
        def Hx2_xla(c,*x,full=False):
                c = _unpack_builder(c)
                x_shape = c.get_shape(x[0])
                dims = x_shape.dimensions()
                dtype = x_shape.element_type()
                dim0 = dims[0]
                if full:
                        dim1 = self.basisClass.numBasisFuncFull
                else:
                        dim1 = self.basisClass.numBasisFunc
                return xla_client.ops.CustomCall(c, xlaName, (_constant_s32_scalar(c,self.basisClass.identifier),
                                                              xla_client.ops.ConcatInDim(c,x,0),
                                                              _constant_array(c,np.array([2],dtype=np.int32)),
                                                              _constant_s32_scalar(c,1),
                                                              _constant_bool(c,full),
                                                              _constant_s32_scalar(c,dim0),
                                                              _constant_s32_scalar(c,dim1)
                                                         ),
                                                         xla_client.Shape.array_shape(dtype,(dim0,dim1)))
        def Hy2_xla(c,*x,full=False):
                c = _unpack_builder(c)
                x_shape = c.get_shape(x[0])
                dims = x_shape.dimensions()
                dtype = x_shape.element_type()
                dim0 = dims[0]
                if full:
                        dim1 = self.basisClass.numBasisFuncFull
                else:
                        dim1 = self.basisClass.numBasisFunc
                return xla_client.ops.CustomCall(c, xlaName, (_constant_s32_scalar(c,self.basisClass.identifier),
                                                              xla_client.ops.ConcatInDim(c,x,0),
                                                              _constant_array(c,np.array([0,2],dtype=np.int32)),
                                                              _constant_s32_scalar(c,2),
                                                              _constant_bool(c,full),
                                                              _constant_s32_scalar(c,dim0),
                                                              _constant_s32_scalar(c,dim1)
                                                         ),
                                                         xla_client.Shape.array_shape(dtype,(dim0,dim1)))
        def Hz2_xla(c,*x,full=False):
                c = _unpack_builder(c)
                x_shape = c.get_shape(x[0])
                dims = x_shape.dimensions()
                dtype = x_shape.element_type()
                dim0 = dims[0]
                if full:
                        dim1 = self.basisClass.numBasisFuncFull
                else:
                        dim1 = self.basisClass.numBasisFunc
                return xla_client.ops.CustomCall(c, xlaName, (_constant_s32_scalar(c,self.basisClass.identifier),
                                                              xla_client.ops.ConcatInDim(c,x,0),
                                                              _constant_array(c,np.array([0,0,2],dtype=np.int32)),
                                                              _constant_s32_scalar(c,3),
                                                              _constant_bool(c,full),
                                                              _constant_s32_scalar(c,dim0),
                                                              _constant_s32_scalar(c,dim1)
                                                         ),
                                                         xla_client.Shape.array_shape(dtype,(dim0,dim1)))
        def Hw2_xla(c,*x,full=False):
                c = _unpack_builder(c)
                x_shape = c.get_shape(x[0])
                dims = x_shape.dimensions()
                dtype = x_shape.element_type()
                dim0 = dims[0]
                if full:
                        dim1 = self.basisClass.numBasisFuncFull
                else:
                        dim1 = self.basisClass.numBasisFunc
                return xla_client.ops.CustomCall(c, xlaName, (_constant_s32_scalar(c,self.basisClass.identifier),
                                                              xla_client.ops.ConcatInDim(c,x,0),
                                                              _constant_array(c,np.array([0,0,0,2],dtype=np.int32)),
                                                              _constant_s32_scalar(c,4),
                                                              _constant_bool(c,full),
                                                              _constant_s32_scalar(c,dim0),
                                                              _constant_s32_scalar(c,dim1)
                                                         ),
                                                         xla_client.Shape.array_shape(dtype,(dim0,dim1)))
        def Hy3_xla(c,*x,full=False):
                c = _unpack_builder(c)
                x_shape = c.get_shape(x[0])
                dims = x_shape.dimensions()
                dtype = x_shape.element_type()
                dim0 = dims[0]
                if full:
                        dim1 = self.basisClass.numBasisFuncFull
                else:
                        dim1 = self.basisClass.numBasisFunc
                return xla_client.ops.CustomCall(c, xlaName, (_constant_s32_scalar(c,self.basisClass.identifier),
                                                              xla_client.ops.ConcatInDim(c,x,0),
                                                              _constant_array(c,np.array([0,3],dtype=np.int32)),
                                                              _constant_s32_scalar(c,2),
                                                              _constant_bool(c,full),
                                                              _constant_s32_scalar(c,dim0),
                                                              _constant_s32_scalar(c,dim1)
                                                         ),
                                                         xla_client.Shape.array_shape(dtype,(dim0,dim1)))
        def Hxy2_xla(c,*x,full=False):
                c = _unpack_builder(c)
                x_shape = c.get_shape(x[0])
                dims = x_shape.dimensions()
                dtype = x_shape.element_type()
                dim0 = dims[0]
                if full:
                        dim1 = self.basisClass.numBasisFuncFull
                else:
                        dim1 = self.basisClass.numBasisFunc
                return xla_client.ops.CustomCall(c, xlaName, (_constant_s32_scalar(c,self.basisClass.identifier),
                                                              xla_client.ops.ConcatInDim(c,x,0),
                                                              _constant_array(c,np.array([1,2],dtype=np.int32)),
                                                              _constant_s32_scalar(c,2),
                                                              _constant_bool(c,full),
                                                              _constant_s32_scalar(c,dim0),
                                                              _constant_s32_scalar(c,dim1)
                                                         ),
                                                         xla_client.Shape.array_shape(dtype,(dim0,dim1)))
        def Hx2y_xla(c,*x,full=False):
                c = _unpack_builder(c)
                x_shape = c.get_shape(x[0])
                dims = x_shape.dimensions()
                dtype = x_shape.element_type()
                dim0 = dims[0]
                if full:
                        dim1 = self.basisClass.numBasisFuncFull
                else:
                        dim1 = self.basisClass.numBasisFunc
                return xla_client.ops.CustomCall(c, xlaName, (_constant_s32_scalar(c,self.basisClass.identifier),
                                                              xla_client.ops.ConcatInDim(c,x,0),
                                                              _constant_array(c,np.array([2,1],dtype=np.int32)),
                                                              _constant_s32_scalar(c,2),
                                                              _constant_bool(c,full),
                                                              _constant_s32_scalar(c,dim0),
                                                              _constant_s32_scalar(c,dim1)
                                                         ),
                                                         xla_client.Shape.array_shape(dtype,(dim0,dim1)))
        def Hx2z_xla(c,*x,full=False):
                c = _unpack_builder(c)
                x_shape = c.get_shape(x[0])
                dims = x_shape.dimensions()
                dtype = x_shape.element_type()
                dim0 = dims[0]
                if full:
                        dim1 = self.basisClass.numBasisFuncFull
                else:
                        dim1 = self.basisClass.numBasisFunc
                return xla_client.ops.CustomCall(c, xlaName, (_constant_s32_scalar(c,self.basisClass.identifier),
                                                              xla_client.ops.ConcatInDim(c,x,0),
                                                              _constant_array(c,np.array([2,0,1],dtype=np.int32)),
                                                              _constant_s32_scalar(c,3),
                                                              _constant_bool(c,full),
                                                              _constant_s32_scalar(c,dim0),
                                                              _constant_s32_scalar(c,dim1)
                                                         ),
                                                         xla_client.Shape.array_shape(dtype,(dim0,dim1)))
        def Hx3_xla(c,*x,full=False):
                c = _unpack_builder(c)
                x_shape = c.get_shape(x[0])
                dims = x_shape.dimensions()
                dtype = x_shape.element_type()
                dim0 = dims[0]
                if full:
                        dim1 = self.basisClass.numBasisFuncFull
                else:
                        dim1 = self.basisClass.numBasisFunc
                return xla_client.ops.CustomCall(c, xlaName, (_constant_s32_scalar(c,self.basisClass.identifier),
                                                              xla_client.ops.ConcatInDim(c,x,0),
                                                              _constant_array(c,np.array([3],dtype=np.int32)),
                                                              _constant_s32_scalar(c,1),
                                                              _constant_bool(c,full),
                                                              _constant_s32_scalar(c,dim0),
                                                              _constant_s32_scalar(c,dim1)
                                                         ),
                                                         xla_client.Shape.array_shape(dtype,(dim0,dim1)))
        def Hz3_xla(c,*x,full=False):
                c = _unpack_builder(c)
                x_shape = c.get_shape(x[0])
                dims = x_shape.dimensions()
                dtype = x_shape.element_type()
                dim0 = dims[0]
                if full:
                        dim1 = self.basisClass.numBasisFuncFull
                else:
                        dim1 = self.basisClass.numBasisFunc
                return xla_client.ops.CustomCall(c, xlaName, (_constant_s32_scalar(c,self.basisClass.identifier),
                                                              xla_client.ops.ConcatInDim(c,x,0),
                                                              _constant_array(c,np.array([0,0,3],dtype=np.int32)),
                                                              _constant_s32_scalar(c,3),
                                                              _constant_bool(c,full),
                                                              _constant_s32_scalar(c,dim0),
                                                              _constant_s32_scalar(c,dim1)
                                                         ),
                                                         xla_client.Shape.array_shape(dtype,(dim0,dim1)))
        def Hy2z_xla(c,*x,full=False):
                c = _unpack_builder(c)
                x_shape = c.get_shape(x[0])
                dims = x_shape.dimensions()
                dtype = x_shape.element_type()
                dim0 = dims[0]
                if full:
                        dim1 = self.basisClass.numBasisFuncFull
                else:
                        dim1 = self.basisClass.numBasisFunc
                return xla_client.ops.CustomCall(c, xlaName, (_constant_s32_scalar(c,self.basisClass.identifier),
                                                              xla_client.ops.ConcatInDim(c,x,0),
                                                              _constant_array(c,np.array([0,2,1],dtype=np.int32)),
                                                              _constant_s32_scalar(c,3),
                                                              _constant_bool(c,full),
                                                              _constant_s32_scalar(c,dim0),
                                                              _constant_s32_scalar(c,dim1)
                                                         ),
                                                         xla_client.Shape.array_shape(dtype,(dim0,dim1)))
        def Hy4_xla(c,*x,full=False):
                c = _unpack_builder(c)
                x_shape = c.get_shape(x[0])
                dims = x_shape.dimensions()
                dtype = x_shape.element_type()
                dim0 = dims[0]
                if full:
                        dim1 = self.basisClass.numBasisFuncFull
                else:
                        dim1 = self.basisClass.numBasisFunc
                return xla_client.ops.CustomCall(c, xlaName, (_constant_s32_scalar(c,self.basisClass.identifier),
                                                              xla_client.ops.ConcatInDim(c,x,0),
                                                              _constant_array(c,np.array([0,4],dtype=np.int32)),
                                                              _constant_s32_scalar(c,2),
                                                              _constant_bool(c,full),
                                                              _constant_s32_scalar(c,dim0),
                                                              _constant_s32_scalar(c,dim1)
                                                         ),
                                                         xla_client.Shape.array_shape(dtype,(dim0,dim1)))
        def Hxy3_xla(c,*x,full=False):
                c = _unpack_builder(c)
                x_shape = c.get_shape(x[0])
                dims = x_shape.dimensions()
                dtype = x_shape.element_type()
                dim0 = dims[0]
                if full:
                        dim1 = self.basisClass.numBasisFuncFull
                else:
                        dim1 = self.basisClass.numBasisFunc
                return xla_client.ops.CustomCall(c, xlaName, (_constant_s32_scalar(c,self.basisClass.identifier),
                                                              xla_client.ops.ConcatInDim(c,x,0),
                                                              _constant_array(c,np.array([1,3],dtype=np.int32)),
                                                              _constant_s32_scalar(c,2),
                                                              _constant_bool(c,full),
                                                              _constant_s32_scalar(c,dim0),
                                                              _constant_s32_scalar(c,dim1)
                                                         ),
                                                         xla_client.Shape.array_shape(dtype,(dim0,dim1)))
        def Hx2y2_xla(c,*x,full=False):
                c = _unpack_builder(c)
                x_shape = c.get_shape(x[0])
                dims = x_shape.dimensions()
                dtype = x_shape.element_type()
                dim0 = dims[0]
                if full:
                        dim1 = self.basisClass.numBasisFuncFull
                else:
                        dim1 = self.basisClass.numBasisFunc
                return xla_client.ops.CustomCall(c, xlaName, (_constant_s32_scalar(c,self.basisClass.identifier),
                                                              xla_client.ops.ConcatInDim(c,x,0),
                                                              _constant_array(c,np.array([2,2],dtype=np.int32)),
                                                              _constant_s32_scalar(c,2),
                                                              _constant_bool(c,full),
                                                              _constant_s32_scalar(c,dim0),
                                                              _constant_s32_scalar(c,dim1)
                                                         ),
                                                         xla_client.Shape.array_shape(dtype,(dim0,dim1)))
        def Hx3y_xla(c,*x,full=False):
                c = _unpack_builder(c)
                x_shape = c.get_shape(x[0])
                dims = x_shape.dimensions()
                dtype = x_shape.element_type()
                dim0 = dims[0]
                if full:
                        dim1 = self.basisClass.numBasisFuncFull
                else:
                        dim1 = self.basisClass.numBasisFunc
                return xla_client.ops.CustomCall(c, xlaName, (_constant_s32_scalar(c,self.basisClass.identifier),
                                                              xla_client.ops.ConcatInDim(c,x,0),
                                                              _constant_array(c,np.array([3,1],dtype=np.int32)),
                                                              _constant_s32_scalar(c,2),
                                                              _constant_bool(c,full),
                                                              _constant_s32_scalar(c,dim0),
                                                              _constant_s32_scalar(c,dim1)
                                                         ),
                                                         xla_client.Shape.array_shape(dtype,(dim0,dim1)))
        def Hx4_xla(c,*x,full=False):
                c = _unpack_builder(c)
                x_shape = c.get_shape(x[0])
                dims = x_shape.dimensions()
                dtype = x_shape.element_type()
                dim0 = dims[0]
                if full:
                        dim1 = self.basisClass.numBasisFuncFull
                else:
                        dim1 = self.basisClass.numBasisFunc
                return xla_client.ops.CustomCall(c, xlaName, (_constant_s32_scalar(c,self.basisClass.identifier),
                                                              xla_client.ops.ConcatInDim(c,x,0),
                                                              _constant_array(c,np.array([4],dtype=np.int32)),
                                                              _constant_s32_scalar(c,1),
                                                              _constant_bool(c,full),
                                                              _constant_s32_scalar(c,dim0),
                                                              _constant_s32_scalar(c,dim1)
                                                         ),
                                                         xla_client.Shape.array_shape(dtype,(dim0,dim1)))
        def Hy5_xla(c,*x,full=False):
                c = _unpack_builder(c)
                x_shape = c.get_shape(x[0])
                dims = x_shape.dimensions()
                dtype = x_shape.element_type()
                dim0 = dims[0]
                if full:
                        dim1 = self.basisClass.numBasisFuncFull
                else:
                        dim1 = self.basisClass.numBasisFunc
                return xla_client.ops.CustomCall(c, xlaName, (_constant_s32_scalar(c,self.basisClass.identifier),
                                                              xla_client.ops.ConcatInDim(c,x,0),
                                                              _constant_array(c,np.array([0,5],dtype=np.int32)),
                                                              _constant_s32_scalar(c,2),
                                                              _constant_bool(c,full),
                                                              _constant_s32_scalar(c,dim0),
                                                              _constant_s32_scalar(c,dim1)
                                                         ),
                                                         xla_client.Shape.array_shape(dtype,(dim0,dim1)))
        def Hxy4_xla(c,*x,full=False):
                c = _unpack_builder(c)
                x_shape = c.get_shape(x[0])
                dims = x_shape.dimensions()
                dtype = x_shape.element_type()
                dim0 = dims[0]
                if full:
                        dim1 = self.basisClass.numBasisFuncFull
                else:
                        dim1 = self.basisClass.numBasisFunc
                return xla_client.ops.CustomCall(c, xlaName, (_constant_s32_scalar(c,self.basisClass.identifier),
                                                              xla_client.ops.ConcatInDim(c,x,0),
                                                              _constant_array(c,np.array([1,4],dtype=np.int32)),
                                                              _constant_s32_scalar(c,2),
                                                              _constant_bool(c,full),
                                                              _constant_s32_scalar(c,dim0),
                                                              _constant_s32_scalar(c,dim1)
                                                         ),
                                                         xla_client.Shape.array_shape(dtype,(dim0,dim1)))
        def Hx2y3_xla(c,*x,full=False):
                c = _unpack_builder(c)
                x_shape = c.get_shape(x[0])
                dims = x_shape.dimensions()
                dtype = x_shape.element_type()
                dim0 = dims[0]
                if full:
                        dim1 = self.basisClass.numBasisFuncFull
                else:
                        dim1 = self.basisClass.numBasisFunc
                return xla_client.ops.CustomCall(c, xlaName, (_constant_s32_scalar(c,self.basisClass.identifier),
                                                              xla_client.ops.ConcatInDim(c,x,0),
                                                              _constant_array(c,np.array([2,3],dtype=np.int32)),
                                                              _constant_s32_scalar(c,2),
                                                              _constant_bool(c,full),
                                                              _constant_s32_scalar(c,dim0),
                                                              _constant_s32_scalar(c,dim1)
                                                         ),
                                                         xla_client.Shape.array_shape(dtype,(dim0,dim1)))
        def Hx3y2_xla(c,*x,full=False):
                c = _unpack_builder(c)
                x_shape = c.get_shape(x[0])
                dims = x_shape.dimensions()
                dtype = x_shape.element_type()
                dim0 = dims[0]
                if full:
                        dim1 = self.basisClass.numBasisFuncFull
                else:
                        dim1 = self.basisClass.numBasisFunc
                return xla_client.ops.CustomCall(c, xlaName, (_constant_s32_scalar(c,self.basisClass.identifier),
                                                              xla_client.ops.ConcatInDim(c,x,0),
                                                              _constant_array(c,np.array([3,2],dtype=np.int32)),
                                                              _constant_s32_scalar(c,2),
                                                              _constant_bool(c,full),
                                                              _constant_s32_scalar(c,dim0),
                                                              _constant_s32_scalar(c,dim1)
                                                         ),
                                                         xla_client.Shape.array_shape(dtype,(dim0,dim1)))
        def Hx4y_xla(c,*x,full=False):
                c = _unpack_builder(c)
                x_shape = c.get_shape(x[0])
                dims = x_shape.dimensions()
                dtype = x_shape.element_type()
                dim0 = dims[0]
                if full:
                        dim1 = self.basisClass.numBasisFuncFull
                else:
                        dim1 = self.basisClass.numBasisFunc
                return xla_client.ops.CustomCall(c, xlaName, (_constant_s32_scalar(c,self.basisClass.identifier),
                                                              xla_client.ops.ConcatInDim(c,x,0),
                                                              _constant_array(c,np.array([4,1],dtype=np.int32)),
                                                              _constant_s32_scalar(c,2),
                                                              _constant_bool(c,full),
                                                              _constant_s32_scalar(c,dim0),
                                                              _constant_s32_scalar(c,dim1)
                                                         ),
                                                         xla_client.Shape.array_shape(dtype,(dim0,dim1)))
        def Hx5_xla(c,*x,full=False):
                c = _unpack_builder(c)
                x_shape = c.get_shape(x[0])
                dims = x_shape.dimensions()
                dtype = x_shape.element_type()
                dim0 = dims[0]
                if full:
                        dim1 = self.basisClass.numBasisFuncFull
                else:
                        dim1 = self.basisClass.numBasisFunc
                return xla_client.ops.CustomCall(c, xlaName, (_constant_s32_scalar(c,self.basisClass.identifier),
                                                              xla_client.ops.ConcatInDim(c,x,0),
                                                              _constant_array(c,np.array([5],dtype=np.int32)),
                                                              _constant_s32_scalar(c,1),
                                                              _constant_bool(c,full),
                                                              _constant_s32_scalar(c,dim0),
                                                              _constant_s32_scalar(c,dim1)
                                                         ),
                                                         xla_client.Shape.array_shape(dtype,(dim0,dim1)))
        def Hy6_xla(c,*x,full=False):
                c = _unpack_builder(c)
                x_shape = c.get_shape(x[0])
                dims = x_shape.dimensions()
                dtype = x_shape.element_type()
                dim0 = dims[0]
                if full:
                        dim1 = self.basisClass.numBasisFuncFull
                else:
                        dim1 = self.basisClass.numBasisFunc
                return xla_client.ops.CustomCall(c, xlaName, (_constant_s32_scalar(c,self.basisClass.identifier),
                                                              xla_client.ops.ConcatInDim(c,x,0),
                                                              _constant_array(c,np.array([0,6],dtype=np.int32)),
                                                              _constant_s32_scalar(c,2),
                                                              _constant_bool(c,full),
                                                              _constant_s32_scalar(c,dim0),
                                                              _constant_s32_scalar(c,dim1)
                                                         ),
                                                         xla_client.Shape.array_shape(dtype,(dim0,dim1)))
        def Hxy5_xla(c,*x,full=False):
                c = _unpack_builder(c)
                x_shape = c.get_shape(x[0])
                dims = x_shape.dimensions()
                dtype = x_shape.element_type()
                dim0 = dims[0]
                if full:
                        dim1 = self.basisClass.numBasisFuncFull
                else:
                        dim1 = self.basisClass.numBasisFunc
                return xla_client.ops.CustomCall(c, xlaName, (_constant_s32_scalar(c,self.basisClass.identifier),
                                                              xla_client.ops.ConcatInDim(c,x,0),
                                                              _constant_array(c,np.array([1,5],dtype=np.int32)),
                                                              _constant_s32_scalar(c,2),
                                                              _constant_bool(c,full),
                                                              _constant_s32_scalar(c,dim0),
                                                              _constant_s32_scalar(c,dim1)
                                                         ),
                                                         xla_client.Shape.array_shape(dtype,(dim0,dim1)))
        def Hx2y4_xla(c,*x,full=False):
                c = _unpack_builder(c)
                x_shape = c.get_shape(x[0])
                dims = x_shape.dimensions()
                dtype = x_shape.element_type()
                dim0 = dims[0]
                if full:
                        dim1 = self.basisClass.numBasisFuncFull
                else:
                        dim1 = self.basisClass.numBasisFunc
                return xla_client.ops.CustomCall(c, xlaName, (_constant_s32_scalar(c,self.basisClass.identifier),
                                                              xla_client.ops.ConcatInDim(c,x,0),
                                                              _constant_array(c,np.array([2,4],dtype=np.int32)),
                                                              _constant_s32_scalar(c,2),
                                                              _constant_bool(c,full),
                                                              _constant_s32_scalar(c,dim0),
                                                              _constant_s32_scalar(c,dim1)
                                                         ),
                                                         xla_client.Shape.array_shape(dtype,(dim0,dim1)))
        def Hx3y3_xla(c,*x,full=False):
                c = _unpack_builder(c)
                x_shape = c.get_shape(x[0])
                dims = x_shape.dimensions()
                dtype = x_shape.element_type()
                dim0 = dims[0]
                if full:
                        dim1 = self.basisClass.numBasisFuncFull
                else:
                        dim1 = self.basisClass.numBasisFunc
                return xla_client.ops.CustomCall(c, xlaName, (_constant_s32_scalar(c,self.basisClass.identifier),
                                                              xla_client.ops.ConcatInDim(c,x,0),
                                                              _constant_array(c,np.array([3,3],dtype=np.int32)),
                                                              _constant_s32_scalar(c,2),
                                                              _constant_bool(c,full),
                                                              _constant_s32_scalar(c,dim0),
                                                              _constant_s32_scalar(c,dim1)
                                                         ),
                                                         xla_client.Shape.array_shape(dtype,(dim0,dim1)))
        def Hx4y2_xla(c,*x,full=False):
                c = _unpack_builder(c)
                x_shape = c.get_shape(x[0])
                dims = x_shape.dimensions()
                dtype = x_shape.element_type()
                dim0 = dims[0]
                if full:
                        dim1 = self.basisClass.numBasisFuncFull
                else:
                        dim1 = self.basisClass.numBasisFunc
                return xla_client.ops.CustomCall(c, xlaName, (_constant_s32_scalar(c,self.basisClass.identifier),
                                                              xla_client.ops.ConcatInDim(c,x,0),
                                                              _constant_array(c,np.array([4,2],dtype=np.int32)),
                                                              _constant_s32_scalar(c,2),
                                                              _constant_bool(c,full),
                                                              _constant_s32_scalar(c,dim0),
                                                              _constant_s32_scalar(c,dim1)
                                                         ),
                                                         xla_client.Shape.array_shape(dtype,(dim0,dim1)))
        def Hx5y_xla(c,*x,full=False):
                c = _unpack_builder(c)
                x_shape = c.get_shape(x[0])
                dims = x_shape.dimensions()
                dtype = x_shape.element_type()
                dim0 = dims[0]
                if full:
                        dim1 = self.basisClass.numBasisFuncFull
                else:
                        dim1 = self.basisClass.numBasisFunc
                return xla_client.ops.CustomCall(c, xlaName, (_constant_s32_scalar(c,self.basisClass.identifier),
                                                              xla_client.ops.ConcatInDim(c,x,0),
                                                              _constant_array(c,np.array([5,1],dtype=np.int32)),
                                                              _constant_s32_scalar(c,2),
                                                              _constant_bool(c,full),
                                                              _constant_s32_scalar(c,dim0),
                                                              _constant_s32_scalar(c,dim1)
                                                         ),
                                                         xla_client.Shape.array_shape(dtype,(dim0,dim1)))
        def Hx6_xla(c,*x,full=False):
                c = _unpack_builder(c)
                x_shape = c.get_shape(x[0])
                dims = x_shape.dimensions()
                dtype = x_shape.element_type()
                dim0 = dims[0]
                if full:
                        dim1 = self.basisClass.numBasisFuncFull
                else:
                        dim1 = self.basisClass.numBasisFunc
                return xla_client.ops.CustomCall(c, xlaName, (_constant_s32_scalar(c,self.basisClass.identifier),
                                                              xla_client.ops.ConcatInDim(c,x,0),
                                                              _constant_array(c,np.array([6],dtype=np.int32)),
                                                              _constant_s32_scalar(c,1),
                                                              _constant_bool(c,full),
                                                              _constant_s32_scalar(c,dim0),
                                                              _constant_s32_scalar(c,dim1)
                                                         ),
                                                         xla_client.Shape.array_shape(dtype,(dim0,dim1)))
        def Hy7_xla(c,*x,full=False):
                c = _unpack_builder(c)
                x_shape = c.get_shape(x[0])
                dims = x_shape.dimensions()
                dtype = x_shape.element_type()
                dim0 = dims[0]
                if full:
                        dim1 = self.basisClass.numBasisFuncFull
                else:
                        dim1 = self.basisClass.numBasisFunc
                return xla_client.ops.CustomCall(c, xlaName, (_constant_s32_scalar(c,self.basisClass.identifier),
                                                              xla_client.ops.ConcatInDim(c,x,0),
                                                              _constant_array(c,np.array([0,7],dtype=np.int32)),
                                                              _constant_s32_scalar(c,2),
                                                              _constant_bool(c,full),
                                                              _constant_s32_scalar(c,dim0),
                                                              _constant_s32_scalar(c,dim1)
                                                         ),
                                                         xla_client.Shape.array_shape(dtype,(dim0,dim1)))
        def Hxy6_xla(c,*x,full=False):
                c = _unpack_builder(c)
                x_shape = c.get_shape(x[0])
                dims = x_shape.dimensions()
                dtype = x_shape.element_type()
                dim0 = dims[0]
                if full:
                        dim1 = self.basisClass.numBasisFuncFull
                else:
                        dim1 = self.basisClass.numBasisFunc
                return xla_client.ops.CustomCall(c, xlaName, (_constant_s32_scalar(c,self.basisClass.identifier),
                                                              xla_client.ops.ConcatInDim(c,x,0),
                                                              _constant_array(c,np.array([1,6],dtype=np.int32)),
                                                              _constant_s32_scalar(c,2),
                                                              _constant_bool(c,full),
                                                              _constant_s32_scalar(c,dim0),
                                                              _constant_s32_scalar(c,dim1)
                                                         ),
                                                         xla_client.Shape.array_shape(dtype,(dim0,dim1)))
        def Hx2y5_xla(c,*x,full=False):
                c = _unpack_builder(c)
                x_shape = c.get_shape(x[0])
                dims = x_shape.dimensions()
                dtype = x_shape.element_type()
                dim0 = dims[0]
                if full:
                        dim1 = self.basisClass.numBasisFuncFull
                else:
                        dim1 = self.basisClass.numBasisFunc
                return xla_client.ops.CustomCall(c, xlaName, (_constant_s32_scalar(c,self.basisClass.identifier),
                                                              xla_client.ops.ConcatInDim(c,x,0),
                                                              _constant_array(c,np.array([2,5],dtype=np.int32)),
                                                              _constant_s32_scalar(c,2),
                                                              _constant_bool(c,full),
                                                              _constant_s32_scalar(c,dim0),
                                                              _constant_s32_scalar(c,dim1)
                                                         ),
                                                         xla_client.Shape.array_shape(dtype,(dim0,dim1)))
        def Hx3y4_xla(c,*x,full=False):
                c = _unpack_builder(c)
                x_shape = c.get_shape(x[0])
                dims = x_shape.dimensions()
                dtype = x_shape.element_type()
                dim0 = dims[0]
                if full:
                        dim1 = self.basisClass.numBasisFuncFull
                else:
                        dim1 = self.basisClass.numBasisFunc
                return xla_client.ops.CustomCall(c, xlaName, (_constant_s32_scalar(c,self.basisClass.identifier),
                                                              xla_client.ops.ConcatInDim(c,x,0),
                                                              _constant_array(c,np.array([3,4],dtype=np.int32)),
                                                              _constant_s32_scalar(c,2),
                                                              _constant_bool(c,full),
                                                              _constant_s32_scalar(c,dim0),
                                                              _constant_s32_scalar(c,dim1)
                                                         ),
                                                         xla_client.Shape.array_shape(dtype,(dim0,dim1)))
        def Hx4y3_xla(c,*x,full=False):
                c = _unpack_builder(c)
                x_shape = c.get_shape(x[0])
                dims = x_shape.dimensions()
                dtype = x_shape.element_type()
                dim0 = dims[0]
                if full:
                        dim1 = self.basisClass.numBasisFuncFull
                else:
                        dim1 = self.basisClass.numBasisFunc
                return xla_client.ops.CustomCall(c, xlaName, (_constant_s32_scalar(c,self.basisClass.identifier),
                                                              xla_client.ops.ConcatInDim(c,x,0),
                                                              _constant_array(c,np.array([4,3],dtype=np.int32)),
                                                              _constant_s32_scalar(c,2),
                                                              _constant_bool(c,full),
                                                              _constant_s32_scalar(c,dim0),
                                                              _constant_s32_scalar(c,dim1)
                                                         ),
                                                         xla_client.Shape.array_shape(dtype,(dim0,dim1)))
        def Hx5y2_xla(c,*x,full=False):
                c = _unpack_builder(c)
                x_shape = c.get_shape(x[0])
                dims = x_shape.dimensions()
                dtype = x_shape.element_type()
                dim0 = dims[0]
                if full:
                        dim1 = self.basisClass.numBasisFuncFull
                else:
                        dim1 = self.basisClass.numBasisFunc
                return xla_client.ops.CustomCall(c, xlaName, (_constant_s32_scalar(c,self.basisClass.identifier),
                                                              xla_client.ops.ConcatInDim(c,x,0),
                                                              _constant_array(c,np.array([5,2],dtype=np.int32)),
                                                              _constant_s32_scalar(c,2),
                                                              _constant_bool(c,full),
                                                              _constant_s32_scalar(c,dim0),
                                                              _constant_s32_scalar(c,dim1)
                                                         ),
                                                         xla_client.Shape.array_shape(dtype,(dim0,dim1)))
        def Hx6y_xla(c,*x,full=False):
                c = _unpack_builder(c)
                x_shape = c.get_shape(x[0])
                dims = x_shape.dimensions()
                dtype = x_shape.element_type()
                dim0 = dims[0]
                if full:
                        dim1 = self.basisClass.numBasisFuncFull
                else:
                        dim1 = self.basisClass.numBasisFunc
                return xla_client.ops.CustomCall(c, xlaName, (_constant_s32_scalar(c,self.basisClass.identifier),
                                                              xla_client.ops.ConcatInDim(c,x,0),
                                                              _constant_array(c,np.array([6,1],dtype=np.int32)),
                                                              _constant_s32_scalar(c,2),
                                                              _constant_bool(c,full),
                                                              _constant_s32_scalar(c,dim0),
                                                              _constant_s32_scalar(c,dim1)
                                                         ),
                                                         xla_client.Shape.array_shape(dtype,(dim0,dim1)))
        def Hx3y5_xla(c,*x,full=False):
                c = _unpack_builder(c)
                x_shape = c.get_shape(x[0])
                dims = x_shape.dimensions()
                dtype = x_shape.element_type()
                dim0 = dims[0]
                if full:
                        dim1 = self.basisClass.numBasisFuncFull
                else:
                        dim1 = self.basisClass.numBasisFunc
                return xla_client.ops.CustomCall(c, xlaName, (_constant_s32_scalar(c,self.basisClass.identifier),
                                                              xla_client.ops.ConcatInDim(c,x,0),
                                                              _constant_array(c,np.array([3,5],dtype=np.int32)),
                                                              _constant_s32_scalar(c,2),
                                                              _constant_bool(c,full),
                                                              _constant_s32_scalar(c,dim0),
                                                              _constant_s32_scalar(c,dim1)
                                                         ),
                                                         xla_client.Shape.array_shape(dtype,(dim0,dim1)))
        def Hx5y3_xla(c,*x,full=False):
                c = _unpack_builder(c)
                x_shape = c.get_shape(x[0])
                dims = x_shape.dimensions()
                dtype = x_shape.element_type()
                dim0 = dims[0]
                if full:
                        dim1 = self.basisClass.numBasisFuncFull
                else:
                        dim1 = self.basisClass.numBasisFunc
                return xla_client.ops.CustomCall(c, xlaName, (_constant_s32_scalar(c,self.basisClass.identifier),
                                                              xla_client.ops.ConcatInDim(c,x,0),
                                                              _constant_array(c,np.array([5,3],dtype=np.int32)),
                                                              _constant_s32_scalar(c,2),
                                                              _constant_bool(c,full),
                                                              _constant_s32_scalar(c,dim0),
                                                              _constant_s32_scalar(c,dim1)
                                                         ),
                                                         xla_client.Shape.array_shape(dtype,(dim0,dim1)))
        def Hx6y2_xla(c,*x,full=False):
                c = _unpack_builder(c)
                x_shape = c.get_shape(x[0])
                dims = x_shape.dimensions()
                dtype = x_shape.element_type()
                dim0 = dims[0]
                if full:
                        dim1 = self.basisClass.numBasisFuncFull
                else:
                        dim1 = self.basisClass.numBasisFunc
                return xla_client.ops.CustomCall(c, xlaName, (_constant_s32_scalar(c,self.basisClass.identifier),
                                                              xla_client.ops.ConcatInDim(c,x,0),
                                                              _constant_array(c,np.array([6,2],dtype=np.int32)),
                                                              _constant_s32_scalar(c,2),
                                                              _constant_bool(c,full),
                                                              _constant_s32_scalar(c,dim0),
                                                              _constant_s32_scalar(c,dim1)
                                                         ),
                                                         xla_client.Shape.array_shape(dtype,(dim0,dim1)))
        def Hx2y6_xla(c,*x,full=False):
                c = _unpack_builder(c)
                x_shape = c.get_shape(x[0])
                dims = x_shape.dimensions()
                dtype = x_shape.element_type()
                dim0 = dims[0]
                if full:
                        dim1 = self.basisClass.numBasisFuncFull
                else:
                        dim1 = self.basisClass.numBasisFunc
                return xla_client.ops.CustomCall(c, xlaName, (_constant_s32_scalar(c,self.basisClass.identifier),
                                                              xla_client.ops.ConcatInDim(c,x,0),
                                                              _constant_array(c,np.array([2,6],dtype=np.int32)),
                                                              _constant_s32_scalar(c,2),
                                                              _constant_bool(c,full),
                                                              _constant_s32_scalar(c,dim0),
                                                              _constant_s32_scalar(c,dim1)
                                                         ),
                                                         xla_client.Shape.array_shape(dtype,(dim0,dim1)))
        def Hx4y4_xla(c,*x,full=False):
                c = _unpack_builder(c)
                x_shape = c.get_shape(x[0])
                dims = x_shape.dimensions()
                dtype = x_shape.element_type()
                dim0 = dims[0]
                if full:
                        dim1 = self.basisClass.numBasisFuncFull
                else:
                        dim1 = self.basisClass.numBasisFunc
                return xla_client.ops.CustomCall(c, xlaName, (_constant_s32_scalar(c,self.basisClass.identifier),
                                                              xla_client.ops.ConcatInDim(c,x,0),
                                                              _constant_array(c,np.array([4,4],dtype=np.int32)),
                                                              _constant_s32_scalar(c,2),
                                                              _constant_bool(c,full),
                                                              _constant_s32_scalar(c,dim0),
                                                              _constant_s32_scalar(c,dim1)
                                                         ),
                                                         xla_client.Shape.array_shape(dtype,(dim0,dim1)))
        def Hx2y7_xla(c,*x,full=False):
                c = _unpack_builder(c)
                x_shape = c.get_shape(x[0])
                dims = x_shape.dimensions()
                dtype = x_shape.element_type()
                dim0 = dims[0]
                if full:
                        dim1 = self.basisClass.numBasisFuncFull
                else:
                        dim1 = self.basisClass.numBasisFunc
                return xla_client.ops.CustomCall(c, xlaName, (_constant_s32_scalar(c,self.basisClass.identifier),
                                                              xla_client.ops.ConcatInDim(c,x,0),
                                                              _constant_array(c,np.array([2,7],dtype=np.int32)),
                                                              _constant_s32_scalar(c,2),
                                                              _constant_bool(c,full),
                                                              _constant_s32_scalar(c,dim0),
                                                              _constant_s32_scalar(c,dim1)
                                                         ),
                                                         xla_client.Shape.array_shape(dtype,(dim0,dim1)))
        def Hx6y3_xla(c,*x,full=False):
                c = _unpack_builder(c)
                x_shape = c.get_shape(x[0])
                dims = x_shape.dimensions()
                dtype = x_shape.element_type()
                dim0 = dims[0]
                if full:
                        dim1 = self.basisClass.numBasisFuncFull
                else:
                        dim1 = self.basisClass.numBasisFunc
                return xla_client.ops.CustomCall(c, xlaName, (_constant_s32_scalar(c,self.basisClass.identifier),
                                                              xla_client.ops.ConcatInDim(c,x,0),
                                                              _constant_array(c,np.array([6,3],dtype=np.int32)),
                                                              _constant_s32_scalar(c,2),
                                                              _constant_bool(c,full),
                                                              _constant_s32_scalar(c,dim0),
                                                              _constant_s32_scalar(c,dim1)
                                                         ),
                                                         xla_client.Shape.array_shape(dtype,(dim0,dim1)))
        def Hx5y4_xla(c,*x,full=False):
                c = _unpack_builder(c)
                x_shape = c.get_shape(x[0])
                dims = x_shape.dimensions()
                dtype = x_shape.element_type()
                dim0 = dims[0]
                if full:
                        dim1 = self.basisClass.numBasisFuncFull
                else:
                        dim1 = self.basisClass.numBasisFunc
                return xla_client.ops.CustomCall(c, xlaName, (_constant_s32_scalar(c,self.basisClass.identifier),
                                                              xla_client.ops.ConcatInDim(c,x,0),
                                                              _constant_array(c,np.array([5,4],dtype=np.int32)),
                                                              _constant_s32_scalar(c,2),
                                                              _constant_bool(c,full),
                                                              _constant_s32_scalar(c,dim0),
                                                              _constant_s32_scalar(c,dim1)
                                                         ),
                                                         xla_client.Shape.array_shape(dtype,(dim0,dim1)))
        def Hx4y5_xla(c,*x,full=False):
                c = _unpack_builder(c)
                x_shape = c.get_shape(x[0])
                dims = x_shape.dimensions()
                dtype = x_shape.element_type()
                dim0 = dims[0]
                if full:
                        dim1 = self.basisClass.numBasisFuncFull
                else:
                        dim1 = self.basisClass.numBasisFunc
                return xla_client.ops.CustomCall(c, xlaName, (_constant_s32_scalar(c,self.basisClass.identifier),
                                                              xla_client.ops.ConcatInDim(c,x,0),
                                                              _constant_array(c,np.array([4,5],dtype=np.int32)),
                                                              _constant_s32_scalar(c,2),
                                                              _constant_bool(c,full),
                                                              _constant_s32_scalar(c,dim0),
                                                              _constant_s32_scalar(c,dim1)
                                                         ),
                                                         xla_client.Shape.array_shape(dtype,(dim0,dim1)))

        xla.backend_specific_translations["cpu"][H_p] = H_xla
        xla.backend_specific_translations["cpu"][Hx_p] = Hx_xla
        xla.backend_specific_translations["cpu"][Hy_p] = Hy_xla
        xla.backend_specific_translations["cpu"][Hz_p] = Hz_xla
        xla.backend_specific_translations["cpu"][Hw_p] = Hw_xla
        xla.backend_specific_translations["cpu"][Hxy_p] = Hxy_xla
        xla.backend_specific_translations["cpu"][Hxz_p] = Hxz_xla
        xla.backend_specific_translations["cpu"][Hxw_p] = Hxw_xla
        xla.backend_specific_translations["cpu"][Hyz_p] = Hyz_xla
        xla.backend_specific_translations["cpu"][Hyw_p] = Hyw_xla
        xla.backend_specific_translations["cpu"][Hzw_p] = Hzw_xla
        xla.backend_specific_translations["cpu"][Hx2_p] = Hx2_xla
        xla.backend_specific_translations["cpu"][Hy2_p] = Hy2_xla
        xla.backend_specific_translations["cpu"][Hz2_p] = Hz2_xla
        xla.backend_specific_translations["cpu"][Hw2_p] = Hw2_xla
        xla.backend_specific_translations["cpu"][Hy3_p] = Hy3_xla
        xla.backend_specific_translations["cpu"][Hxy2_p] = Hxy2_xla
        xla.backend_specific_translations["cpu"][Hx2y_p] = Hx2y_xla
        xla.backend_specific_translations["cpu"][Hx2z_p] = Hx2z_xla
        xla.backend_specific_translations["cpu"][Hx3_p] = Hx3_xla
        xla.backend_specific_translations["cpu"][Hz3_p] = Hz3_xla
        xla.backend_specific_translations["cpu"][Hy2z_p] = Hy2z_xla
        xla.backend_specific_translations["cpu"][Hy4_p] = Hy4_xla
        xla.backend_specific_translations["cpu"][Hxy3_p] = Hxy3_xla
        xla.backend_specific_translations["cpu"][Hx2y2_p] = Hx2y2_xla
        xla.backend_specific_translations["cpu"][Hx3y_p] = Hx3y_xla
        xla.backend_specific_translations["cpu"][Hx4_p] = Hx4_xla
        xla.backend_specific_translations["cpu"][Hy5_p] = Hy5_xla
        xla.backend_specific_translations["cpu"][Hxy4_p] = Hxy4_xla
        xla.backend_specific_translations["cpu"][Hx2y3_p] = Hx2y3_xla
        xla.backend_specific_translations["cpu"][Hx3y2_p] = Hx3y2_xla
        xla.backend_specific_translations["cpu"][Hx4y_p] = Hx4y_xla
        xla.backend_specific_translations["cpu"][Hx5_p] = Hx5_xla
        xla.backend_specific_translations["cpu"][Hy6_p] = Hy6_xla
        xla.backend_specific_translations["cpu"][Hxy5_p] = Hxy5_xla
        xla.backend_specific_translations["cpu"][Hx2y4_p] = Hx2y4_xla
        xla.backend_specific_translations["cpu"][Hx3y3_p] = Hx3y3_xla
        xla.backend_specific_translations["cpu"][Hx4y2_p] = Hx4y2_xla
        xla.backend_specific_translations["cpu"][Hx5y_p] = Hx5y_xla
        xla.backend_specific_translations["cpu"][Hx6_p] = Hx6_xla
        xla.backend_specific_translations["cpu"][Hy7_p] = Hy7_xla
        xla.backend_specific_translations["cpu"][Hxy6_p] = Hxy6_xla
        xla.backend_specific_translations["cpu"][Hx2y5_p] = Hx2y5_xla
        xla.backend_specific_translations["cpu"][Hx3y4_p] = Hx3y4_xla
        xla.backend_specific_translations["cpu"][Hx4y3_p] = Hx4y3_xla
        xla.backend_specific_translations["cpu"][Hx5y2_p] = Hx5y2_xla
        xla.backend_specific_translations["cpu"][Hx6y_p] = Hx6y_xla
        xla.backend_specific_translations["cpu"][Hx3y5_p] = Hx3y5_xla
        xla.backend_specific_translations["cpu"][Hx5y3_p] = Hx5y3_xla
        xla.backend_specific_translations["cpu"][Hx6y2_p] = Hx6y2_xla
        xla.backend_specific_translations["cpu"][Hx2y6_p] = Hx2y6_xla
        xla.backend_specific_translations["cpu"][Hx4y4_p] = Hx4y4_xla
        xla.backend_specific_translations["cpu"][Hx2y7_p] = Hx2y7_xla
        xla.backend_specific_translations["cpu"][Hx6y3_p] = Hx6y3_xla
        xla.backend_specific_translations["cpu"][Hx5y4_p] = Hx5y4_xla
        xla.backend_specific_translations["cpu"][Hx4y5_p] = Hx4y5_xla

        # Batching translations
        def H_batch(vec,batch,full=False):
                return Hjax(*vec,full=full), batch[0]
        def Hx_batch(vec,batch,full=False):
                return Hxjax(*vec,full=full), batch[0]
        def Hy_batch(vec,batch,full=False):
                return Hyjax(*vec,full=full), batch[0]
        def Hz_batch(vec,batch,full=False):
                return Hzjax(*vec,full=full), batch[0]
        def Hw_batch(vec,batch,full=False):
                return Hwjax(*vec,full=full), batch[0]
        def Hxy_batch(vec,batch,full=False):
                return Hxyjax(*vec,full=full), batch[0]
        def Hxz_batch(vec,batch,full=False):
                return Hxzjax(*vec,full=full), batch[0]
        def Hxw_batch(vec,batch,full=False):
                return Hxwjax(*vec,full=full), batch[0]
        def Hyz_batch(vec,batch,full=False):
                return Hyzjax(*vec,full=full), batch[0]
        def Hyw_batch(vec,batch,full=False):
                return Hywjax(*vec,full=full), batch[0]
        def Hzw_batch(vec,batch,full=False):
                return Hzwjax(*vec,full=full), batch[0]
        def Hx2_batch(vec,batch,full=False):
                return Hx2jax(*vec,full=full), batch[0]
        def Hy2_batch(vec,batch,full=False):
                return Hy2jax(*vec,full=full), batch[0]
        def Hz2_batch(vec,batch,full=False):
                return Hz2jax(*vec,full=full), batch[0]
        def Hw2_batch(vec,batch,full=False):
                return Hw2jax(*vec,full=full), batch[0]
        def Hy3_batch(vec,batch,full=False):
                return Hy3jax(*vec,full=full), batch[0]
        def Hxy2_batch(vec,batch,full=False):
                return Hxy2jax(*vec,full=full), batch[0]
        def Hx2y_batch(vec,batch,full=False):
                return Hx2yjax(*vec,full=full), batch[0]
        def Hx2z_batch(vec,batch,full=False):
                return Hx2zjax(*vec,full=full), batch[0]
        def Hx3_batch(vec,batch,full=False):
                return Hx3jax(*vec,full=full), batch[0]
        def Hz3_batch(vec,batch,full=False):
                return Hz3jax(*vec,full=full), batch[0]
        def Hy2z_batch(vec,batch,full=False):
                return Hy2zjax(*vec,full=full), batch[0]
        def Hy4_batch(vec,batch,full=False):
                return Hy4jax(*vec,full=full), batch[0]
        def Hxy3_batch(vec,batch,full=False):
                return Hxy3jax(*vec,full=full), batch[0]
        def Hx2y2_batch(vec,batch,full=False):
                return Hx2y2jax(*vec,full=full), batch[0]
        def Hx3y_batch(vec,batch,full=False):
                return Hx3yjax(*vec,full=full), batch[0]
        def Hx4_batch(vec,batch,full=False):
                return Hx4jax(*vec,full=full), batch[0]
        def Hy5_batch(vec,batch,full=False):
                return Hy5jax(*vec,full=full), batch[0]
        def Hxy4_batch(vec,batch,full=False):
                return Hxy4jax(*vec,full=full), batch[0]
        def Hx2y3_batch(vec,batch,full=False):
                return Hx2y3jax(*vec,full=full), batch[0]
        def Hx3y2_batch(vec,batch,full=False):
                return Hx3y2jax(*vec,full=full), batch[0]
        def Hx4y_batch(vec,batch,full=False):
                return Hx4yjax(*vec,full=full), batch[0]
        def Hx5_batch(vec,batch,full=False):
                return Hx5jax(*vec,full=full), batch[0]
        def Hy6_batch(vec,batch,full=False):
                return Hy6jax(*vec,full=full), batch[0]
        def Hxy5_batch(vec,batch,full=False):
                return Hxy5jax(*vec,full=full), batch[0]
        def Hx2y4_batch(vec,batch,full=False):
                return Hx2y4jax(*vec,full=full), batch[0]
        def Hx3y3_batch(vec,batch,full=False):
                return Hx3y3jax(*vec,full=full), batch[0]
        def Hx4y2_batch(vec,batch,full=False):
                return Hx4y2jax(*vec,full=full), batch[0]
        def Hx5y_batch(vec,batch,full=False):
                return Hx5yjax(*vec,full=full), batch[0]
        def Hx6_batch(vec,batch,full=False):
                return Hx6jax(*vec,full=full), batch[0]
        def Hy7_batch(vec,batch,full=False):
                return Hy7jax(*vec,full=full), batch[0]
        def Hxy6_batch(vec,batch,full=False):
                return Hxy6jax(*vec,full=full), batch[0]
        def Hx2y5_batch(vec,batch,full=False):
                return Hx2y5jax(*vec,full=full), batch[0]
        def Hx3y4_batch(vec,batch,full=False):
                return Hx3y4jax(*vec,full=full), batch[0]
        def Hx4y3_batch(vec,batch,full=False):
                return Hx4y3jax(*vec,full=full), batch[0]
        def Hx5y2_batch(vec,batch,full=False):
                return Hx5y2jax(*vec,full=full), batch[0]
        def Hx6y_batch(vec,batch,full=False):
                return Hx6yjax(*vec,full=full), batch[0]
        def Hx3y5_batch(vec,batch,full=False):
                return Hx3y5jax(*vec,full=full), batch[0]
        def Hx5y3_batch(vec,batch,full=False):
                return Hx5y3jax(*vec,full=full), batch[0]
        def Hx6y2_batch(vec,batch,full=False):
                return Hx6y2jax(*vec,full=full), batch[0]
        def Hx2y6_batch(vec,batch,full=False):
                return Hx2y6jax(*vec,full=full), batch[0]
        def Hx4y4_batch(vec,batch,full=False):
                return Hx4y4jax(*vec,full=full), batch[0]
        def Hx2y7_batch(vec,batch,full=False):
                return Hx2y7jax(*vec,full=full), batch[0]
        def Hx6y3_batch(vec,batch,full=False):
                return Hx6y3jax(*vec,full=full), batch[0]
        def Hx5y4_batch(vec,batch,full=False):
                return Hx5y4jax(*vec,full=full), batch[0]
        def Hx4y5_batch(vec,batch,full=False):
                return Hx4y5jax(*vec,full=full), batch[0]

        batching.primitive_batchers[H_p] = H_batch
        batching.primitive_batchers[Hx_p] = Hx_batch
        batching.primitive_batchers[Hy_p] = Hy_batch
        batching.primitive_batchers[Hz_p] = Hz_batch
        batching.primitive_batchers[Hw_p] = Hw_batch
        batching.primitive_batchers[Hxy_p] = Hxy_batch
        batching.primitive_batchers[Hxz_p] = Hxz_batch
        batching.primitive_batchers[Hxw_p] = Hxw_batch
        batching.primitive_batchers[Hyz_p] = Hyz_batch
        batching.primitive_batchers[Hyw_p] = Hyw_batch
        batching.primitive_batchers[Hzw_p] = Hzw_batch
        batching.primitive_batchers[Hx2_p] = Hx2_batch
        batching.primitive_batchers[Hy2_p] = Hy2_batch
        batching.primitive_batchers[Hz2_p] = Hz2_batch
        batching.primitive_batchers[Hw2_p] = Hw2_batch
        batching.primitive_batchers[Hy3_p] = Hy3_batch
        batching.primitive_batchers[Hxy2_p] = Hxy2_batch
        batching.primitive_batchers[Hx2y_p] = Hx2y_batch
        batching.primitive_batchers[Hx2z_p] = Hx2z_batch
        batching.primitive_batchers[Hx3_p] = Hx3_batch
        batching.primitive_batchers[Hz3_p] = Hz3_batch
        batching.primitive_batchers[Hy2z_p] = Hy2z_batch
        batching.primitive_batchers[Hy4_p] = Hy4_batch
        batching.primitive_batchers[Hxy3_p] = Hxy3_batch
        batching.primitive_batchers[Hx2y2_p] = Hx2y2_batch
        batching.primitive_batchers[Hx3y_p] = Hx3y_batch
        batching.primitive_batchers[Hx4_p] = Hx4_batch
        batching.primitive_batchers[Hy5_p] = Hy5_batch
        batching.primitive_batchers[Hxy4_p] = Hxy4_batch
        batching.primitive_batchers[Hx2y3_p] = Hx2y3_batch
        batching.primitive_batchers[Hx3y2_p] = Hx3y2_batch
        batching.primitive_batchers[Hx4y_p] = Hx4y_batch
        batching.primitive_batchers[Hx5_p] = Hx5_batch
        batching.primitive_batchers[Hy6_p] = Hy6_batch
        batching.primitive_batchers[Hxy5_p] = Hxy5_batch
        batching.primitive_batchers[Hx2y4_p] = Hx2y4_batch
        batching.primitive_batchers[Hx3y3_p] = Hx3y3_batch
        batching.primitive_batchers[Hx4y2_p] = Hx4y2_batch
        batching.primitive_batchers[Hx5y_p] = Hx5y_batch
        batching.primitive_batchers[Hx6_p] = Hx6_batch
        batching.primitive_batchers[Hy7_p] = Hy7_batch
        batching.primitive_batchers[Hxy6_p] = Hxy6_batch
        batching.primitive_batchers[Hx2y5_p] = Hx2y5_batch
        batching.primitive_batchers[Hx3y4_p] = Hx3y4_batch
        batching.primitive_batchers[Hx4y3_p] = Hx4y3_batch
        batching.primitive_batchers[Hx5y2_p] = Hx5y2_batch
        batching.primitive_batchers[Hx6y_p] = Hx6y_batch
        batching.primitive_batchers[Hx3y5_p] = Hx3y5_batch
        batching.primitive_batchers[Hx5y3_p] = Hx5y3_batch
        batching.primitive_batchers[Hx6y2_p] = Hx6y2_batch
        batching.primitive_batchers[Hx2y6_p] = Hx2y6_batch
        batching.primitive_batchers[Hx4y4_p] = Hx4y4_batch
        batching.primitive_batchers[Hx2y7_p] = Hx2y7_batch
        batching.primitive_batchers[Hx6y3_p] = Hx6y3_batch
        batching.primitive_batchers[Hx5y4_p] = Hx5y4_batch
        batching.primitive_batchers[Hx4y5_p] = Hx4y5_batch

        # Jacobian vector translations
        def H_jvp(arg_vals,arg_tans,full=False):
                funcs = [Hxjax,Hyjax,Hzjax,Hwjax]
                n = min(len(arg_vals),len(funcs))
                flat = len(arg_vals[0].shape) == 1
                dim0 = arg_vals[0].shape[0]
                if full:
                        dim1 = self.basisClass.numBasisFuncFull
                else:
                        dim1 = self.basisClass.numBasisFunc
                out_tans = np.zeros((dim0,dim1))
                for k in range(n):
                        if not (type(arg_tans[k]) is ad.Zero):
                                if type(arg_tans[k]) is batching.BatchTracer:
                                        flag = onp.any(arg_tans[k].val != 0)
                                else:
                                        flag = onp.any(arg_tans[k] != 0)
                                if flag:
                                        if flat:
                                                out_tans += funcs[k](*arg_vals,full=full)*np.expand_dims(arg_tans[k],1)
                                        else:
                                                out_tans += funcs[k](*arg_vals,full=full)*arg_tans[k]
                return (Hjax(*arg_vals,full=full),out_tans)
        def Hx_jvp(arg_vals,arg_tans,full=False):
                funcs = [Hx2jax,Hxyjax,Hxzjax,Hxwjax]
                n = min(len(arg_vals),len(funcs))
                flat = len(arg_vals[0].shape) == 1
                dim0 = arg_vals[0].shape[0]
                if full:
                        dim1 = self.basisClass.numBasisFuncFull
                else:
                        dim1 = self.basisClass.numBasisFunc
                out_tans = np.zeros((dim0,dim1))
                for k in range(n):
                        if not (type(arg_tans[k]) is ad.Zero):
                                if type(arg_tans[k]) is batching.BatchTracer:
                                        flag = onp.any(arg_tans[k].val != 0)
                                else:
                                        flag = onp.any(arg_tans[k] != 0)
                                if flag:
                                        if flat:
                                                out_tans += funcs[k](*arg_vals,full=full)*np.expand_dims(arg_tans[k],1)
                                        else:
                                                out_tans += funcs[k](*arg_vals,full=full)*arg_tans[k]
                return (Hxjax(*arg_vals,full=full),out_tans)
        def Hy_jvp(arg_vals,arg_tans,full=False):
                funcs = [Hxyjax,Hy2jax,Hyzjax,Hywjax]
                n = min(len(arg_vals),len(funcs))
                flat = len(arg_vals[0].shape) == 1
                dim0 = arg_vals[0].shape[0]
                if full:
                        dim1 = self.basisClass.numBasisFuncFull
                else:
                        dim1 = self.basisClass.numBasisFunc
                out_tans = np.zeros((dim0,dim1))
                for k in range(n):
                        if not (type(arg_tans[k]) is ad.Zero):
                                if type(arg_tans[k]) is batching.BatchTracer:
                                        flag = onp.any(arg_tans[k].val != 0)
                                else:
                                        flag = onp.any(arg_tans[k] != 0)
                                if flag:
                                        if flat:
                                                out_tans += funcs[k](*arg_vals,full=full)*np.expand_dims(arg_tans[k],1)
                                        else:
                                                out_tans += funcs[k](*arg_vals,full=full)*arg_tans[k]
                return (Hyjax(*arg_vals,full=full),out_tans)
        def Hz_jvp(arg_vals,arg_tans,full=False):
                funcs = [Hxzjax,Hyzjax,Hz2jax,Hzwjax]
                n = min(len(arg_vals),len(funcs))
                flat = len(arg_vals[0].shape) == 1
                dim0 = arg_vals[0].shape[0]
                if full:
                        dim1 = self.basisClass.numBasisFuncFull
                else:
                        dim1 = self.basisClass.numBasisFunc
                out_tans = np.zeros((dim0,dim1))
                for k in range(n):
                        if not (type(arg_tans[k]) is ad.Zero):
                                if type(arg_tans[k]) is batching.BatchTracer:
                                        flag = onp.any(arg_tans[k].val != 0)
                                else:
                                        flag = onp.any(arg_tans[k] != 0)
                                if flag:
                                        if flat:
                                                out_tans += funcs[k](*arg_vals,full=full)*np.expand_dims(arg_tans[k],1)
                                        else:
                                                out_tans += funcs[k](*arg_vals,full=full)*arg_tans[k]
                return (Hzjax(*arg_vals,full=full),out_tans)
        def Hw_jvp(arg_vals,arg_tans,full=False):
                funcs = [Hxwjax,Hywjax,Hzwjax,Hw2jax]
                n = min(len(arg_vals),len(funcs))
                flat = len(arg_vals[0].shape) == 1
                dim0 = arg_vals[0].shape[0]
                if full:
                        dim1 = self.basisClass.numBasisFuncFull
                else:
                        dim1 = self.basisClass.numBasisFunc
                out_tans = np.zeros((dim0,dim1))
                for k in range(n):
                        if not (type(arg_tans[k]) is ad.Zero):
                                if type(arg_tans[k]) is batching.BatchTracer:
                                        flag = onp.any(arg_tans[k].val != 0)
                                else:
                                        flag = onp.any(arg_tans[k] != 0)
                                if flag:
                                        if flat:
                                                out_tans += funcs[k](*arg_vals,full=full)*np.expand_dims(arg_tans[k],1)
                                        else:
                                                out_tans += funcs[k](*arg_vals,full=full)*arg_tans[k]
                return (Hwjax(*arg_vals,full=full),out_tans)
        def Hxy_jvp(arg_vals,arg_tans,full=False):
                funcs = [Hx2yjax,Hxy2jax]
                n = min(len(arg_vals),len(funcs))
                flat = len(arg_vals[0].shape) == 1
                dim0 = arg_vals[0].shape[0]
                if full:
                        dim1 = self.basisClass.numBasisFuncFull
                else:
                        dim1 = self.basisClass.numBasisFunc
                out_tans = np.zeros((dim0,dim1))
                for k in range(n):
                        if not (type(arg_tans[k]) is ad.Zero):
                                if type(arg_tans[k]) is batching.BatchTracer:
                                        flag = onp.any(arg_tans[k].val != 0)
                                else:
                                        flag = onp.any(arg_tans[k] != 0)
                                if flag:
                                        if flat:
                                                out_tans += funcs[k](*arg_vals,full=full)*np.expand_dims(arg_tans[k],1)
                                        else:
                                                out_tans += funcs[k](*arg_vals,full=full)*arg_tans[k]
                return (Hxyjax(*arg_vals,full=full),out_tans)
        def Hxz_jvp(arg_vals,arg_tans,full=False):
                funcs = [Hx2zjax]
                n = min(len(arg_vals),len(funcs))
                flat = len(arg_vals[0].shape) == 1
                dim0 = arg_vals[0].shape[0]
                if full:
                        dim1 = self.basisClass.numBasisFuncFull
                else:
                        dim1 = self.basisClass.numBasisFunc
                out_tans = np.zeros((dim0,dim1))
                for k in range(n):
                        if not (type(arg_tans[k]) is ad.Zero):
                                if type(arg_tans[k]) is batching.BatchTracer:
                                        flag = onp.any(arg_tans[k].val != 0)
                                else:
                                        flag = onp.any(arg_tans[k] != 0)
                                if flag:
                                        if flat:
                                                out_tans += funcs[k](*arg_vals,full=full)*np.expand_dims(arg_tans[k],1)
                                        else:
                                                out_tans += funcs[k](*arg_vals,full=full)*arg_tans[k]
                return (Hxzjax(*arg_vals,full=full),out_tans)
        def Hyz_jvp(arg_vals,arg_tans,full=False):
                funcs = [Hy2zjax]
                n = min(len(arg_vals),len(funcs))
                flat = len(arg_vals[0].shape) == 1
                dim0 = arg_vals[0].shape[0]
                if full:
                        dim1 = self.basisClass.numBasisFuncFull
                else:
                        dim1 = self.basisClass.numBasisFunc
                out_tans = np.zeros((dim0,dim1))
                for k in range(n):
                        if not (type(arg_tans[k]) is ad.Zero):
                                if type(arg_tans[k]) is batching.BatchTracer:
                                        flag = onp.any(arg_tans[k].val != 0)
                                else:
                                        flag = onp.any(arg_tans[k] != 0)
                                if flag:
                                        if flat:
                                                out_tans += funcs[k](*arg_vals,full=full)*np.expand_dims(arg_tans[k],1)
                                        else:
                                                out_tans += funcs[k](*arg_vals,full=full)*arg_tans[k]
                return (Hyzjax(*arg_vals,full=full),out_tans)
        def Hx2_jvp(arg_vals,arg_tans,full=False):
                funcs = [Hx3jax,Hx2yjax]
                n = min(len(arg_vals),len(funcs))
                flat = len(arg_vals[0].shape) == 1
                dim0 = arg_vals[0].shape[0]
                if full:
                        dim1 = self.basisClass.numBasisFuncFull
                else:
                        dim1 = self.basisClass.numBasisFunc
                out_tans = np.zeros((dim0,dim1))
                for k in range(n):
                        if not (type(arg_tans[k]) is ad.Zero):
                                if type(arg_tans[k]) is batching.BatchTracer:
                                        flag = onp.any(arg_tans[k].val != 0)
                                else:
                                        flag = onp.any(arg_tans[k] != 0)
                                if flag:
                                        if flat:
                                                out_tans += funcs[k](*arg_vals,full=full)*np.expand_dims(arg_tans[k],1)
                                        else:
                                                out_tans += funcs[k](*arg_vals,full=full)*arg_tans[k]
                return (Hx2jax(*arg_vals,full=full),out_tans)
        def Hy2_jvp(arg_vals,arg_tans,full=False):
                funcs = [Hxy2jax,Hy3jax]
                n = min(len(arg_vals),len(funcs))
                flat = len(arg_vals[0].shape) == 1
                dim0 = arg_vals[0].shape[0]
                if full:
                        dim1 = self.basisClass.numBasisFuncFull
                else:
                        dim1 = self.basisClass.numBasisFunc
                out_tans = np.zeros((dim0,dim1))
                for k in range(n):
                        if not (type(arg_tans[k]) is ad.Zero):
                                if type(arg_tans[k]) is batching.BatchTracer:
                                        flag = onp.any(arg_tans[k].val != 0)
                                else:
                                        flag = onp.any(arg_tans[k] != 0)
                                if flag:
                                        if flat:
                                                out_tans += funcs[k](*arg_vals,full=full)*np.expand_dims(arg_tans[k],1)
                                        else:
                                                out_tans += funcs[k](*arg_vals,full=full)*arg_tans[k]
                return (Hy2jax(*arg_vals,full=full),out_tans)
        def Hz2_jvp(arg_vals,arg_tans,full=False):
                funcs = [Hz3jax]
                n = min(len(arg_vals),len(funcs))
                flat = len(arg_vals[0].shape) == 1
                dim0 = arg_vals[0].shape[0]
                if full:
                        dim1 = self.basisClass.numBasisFuncFull
                else:
                        dim1 = self.basisClass.numBasisFunc
                out_tans = np.zeros((dim0,dim1))
                for k in range(n):
                        if not (type(arg_tans[k]) is ad.Zero):
                                if type(arg_tans[k]) is batching.BatchTracer:
                                        flag = onp.any(arg_tans[k].val != 0)
                                else:
                                        flag = onp.any(arg_tans[k] != 0)
                                if flag:
                                        if flat:
                                                out_tans += funcs[k](*arg_vals,full=full)*np.expand_dims(arg_tans[k],1)
                                        else:
                                                out_tans += funcs[k](*arg_vals,full=full)*arg_tans[k]
                return (Hz2jax(*arg_vals,full=full),out_tans)
        def Hy3_jvp(arg_vals,arg_tans,full=False):
                funcs = [Hxy3jax,Hy4jax]
                n = min(len(arg_vals),len(funcs))
                flat = len(arg_vals[0].shape) == 1
                dim0 = arg_vals[0].shape[0]
                if full:
                        dim1 = self.basisClass.numBasisFuncFull
                else:
                        dim1 = self.basisClass.numBasisFunc
                out_tans = np.zeros((dim0,dim1))
                for k in range(n):
                        if not (type(arg_tans[k]) is ad.Zero):
                                if type(arg_tans[k]) is batching.BatchTracer:
                                        flag = onp.any(arg_tans[k].val != 0)
                                else:
                                        flag = onp.any(arg_tans[k] != 0)
                                if flag:
                                        if flat:
                                                out_tans += funcs[k](*arg_vals,full=full)*np.expand_dims(arg_tans[k],1)
                                        else:
                                                out_tans += funcs[k](*arg_vals,full=full)*arg_tans[k]
                return (Hy3jax(*arg_vals,full=full),out_tans)
        def Hxy2_jvp(arg_vals,arg_tans,full=False):
                funcs = [Hx2y2jax,Hxy3jax]
                n = min(len(arg_vals),len(funcs))
                flat = len(arg_vals[0].shape) == 1
                dim0 = arg_vals[0].shape[0]
                if full:
                        dim1 = self.basisClass.numBasisFuncFull
                else:
                        dim1 = self.basisClass.numBasisFunc
                out_tans = np.zeros((dim0,dim1))
                for k in range(n):
                        if not (type(arg_tans[k]) is ad.Zero):
                                if type(arg_tans[k]) is batching.BatchTracer:
                                        flag = onp.any(arg_tans[k].val != 0)
                                else:
                                        flag = onp.any(arg_tans[k] != 0)
                                if flag:
                                        if flat:
                                                out_tans += funcs[k](*arg_vals,full=full)*np.expand_dims(arg_tans[k],1)
                                        else:
                                                out_tans += funcs[k](*arg_vals,full=full)*arg_tans[k]
                return (Hxy2jax(*arg_vals,full=full),out_tans)
        def Hx2y_jvp(arg_vals,arg_tans,full=False):
                funcs = [Hx3yjax,Hx2y2jax]
                n = min(len(arg_vals),len(funcs))
                flat = len(arg_vals[0].shape) == 1
                dim0 = arg_vals[0].shape[0]
                if full:
                        dim1 = self.basisClass.numBasisFuncFull
                else:
                        dim1 = self.basisClass.numBasisFunc
                out_tans = np.zeros((dim0,dim1))
                for k in range(n):
                        if not (type(arg_tans[k]) is ad.Zero):
                                if type(arg_tans[k]) is batching.BatchTracer:
                                        flag = onp.any(arg_tans[k].val != 0)
                                else:
                                        flag = onp.any(arg_tans[k] != 0)
                                if flag:
                                        if flat:
                                                out_tans += funcs[k](*arg_vals,full=full)*np.expand_dims(arg_tans[k],1)
                                        else:
                                                out_tans += funcs[k](*arg_vals,full=full)*arg_tans[k]
                return (Hx2yjax(*arg_vals,full=full),out_tans)
        def Hx3_jvp(arg_vals,arg_tans,full=False):
                funcs = [Hx4jax,Hx3yjax]
                n = min(len(arg_vals),len(funcs))
                flat = len(arg_vals[0].shape) == 1
                dim0 = arg_vals[0].shape[0]
                if full:
                        dim1 = self.basisClass.numBasisFuncFull
                else:
                        dim1 = self.basisClass.numBasisFunc
                out_tans = np.zeros((dim0,dim1))
                for k in range(n):
                        if not (type(arg_tans[k]) is ad.Zero):
                                if type(arg_tans[k]) is batching.BatchTracer:
                                        flag = onp.any(arg_tans[k].val != 0)
                                else:
                                        flag = onp.any(arg_tans[k] != 0)
                                if flag:
                                        if flat:
                                                out_tans += funcs[k](*arg_vals,full=full)*np.expand_dims(arg_tans[k],1)
                                        else:
                                                out_tans += funcs[k](*arg_vals,full=full)*arg_tans[k]
                return (Hx3jax(*arg_vals,full=full),out_tans)
        def Hy4_jvp(arg_vals,arg_tans,full=False):
                funcs = [Hxy4jax,Hy5jax]
                n = min(len(arg_vals),len(funcs))
                flat = len(arg_vals[0].shape) == 1
                dim0 = arg_vals[0].shape[0]
                if full:
                        dim1 = self.basisClass.numBasisFuncFull
                else:
                        dim1 = self.basisClass.numBasisFunc
                out_tans = np.zeros((dim0,dim1))
                for k in range(n):
                        if not (type(arg_tans[k]) is ad.Zero):
                                if type(arg_tans[k]) is batching.BatchTracer:
                                        flag = onp.any(arg_tans[k].val != 0)
                                else:
                                        flag = onp.any(arg_tans[k] != 0)
                                if flag:
                                        if flat:
                                                out_tans += funcs[k](*arg_vals,full=full)*np.expand_dims(arg_tans[k],1)
                                        else:
                                                out_tans += funcs[k](*arg_vals,full=full)*arg_tans[k]
                return (Hy4jax(*arg_vals,full=full),out_tans)
        def Hxy3_jvp(arg_vals,arg_tans,full=False):
                funcs = [Hx2y3jax,Hxy4jax]
                n = min(len(arg_vals),len(funcs))
                flat = len(arg_vals[0].shape) == 1
                dim0 = arg_vals[0].shape[0]
                if full:
                        dim1 = self.basisClass.numBasisFuncFull
                else:
                        dim1 = self.basisClass.numBasisFunc
                out_tans = np.zeros((dim0,dim1))
                for k in range(n):
                        if not (type(arg_tans[k]) is ad.Zero):
                                if type(arg_tans[k]) is batching.BatchTracer:
                                        flag = onp.any(arg_tans[k].val != 0)
                                else:
                                        flag = onp.any(arg_tans[k] != 0)
                                if flag:
                                        if flat:
                                                out_tans += funcs[k](*arg_vals,full=full)*np.expand_dims(arg_tans[k],1)
                                        else:
                                                out_tans += funcs[k](*arg_vals,full=full)*arg_tans[k]
                return (Hxy3jax(*arg_vals,full=full),out_tans)
        def Hx2y2_jvp(arg_vals,arg_tans,full=False):
                funcs = [Hx3y2jax,Hx2y3jax]
                n = min(len(arg_vals),len(funcs))
                flat = len(arg_vals[0].shape) == 1
                dim0 = arg_vals[0].shape[0]
                if full:
                        dim1 = self.basisClass.numBasisFuncFull
                else:
                        dim1 = self.basisClass.numBasisFunc
                out_tans = np.zeros((dim0,dim1))
                for k in range(n):
                        if not (type(arg_tans[k]) is ad.Zero):
                                if type(arg_tans[k]) is batching.BatchTracer:
                                        flag = onp.any(arg_tans[k].val != 0)
                                else:
                                        flag = onp.any(arg_tans[k] != 0)
                                if flag:
                                        if flat:
                                                out_tans += funcs[k](*arg_vals,full=full)*np.expand_dims(arg_tans[k],1)
                                        else:
                                                out_tans += funcs[k](*arg_vals,full=full)*arg_tans[k]
                return (Hx2y2jax(*arg_vals,full=full),out_tans)
        def Hx3y_jvp(arg_vals,arg_tans,full=False):
                funcs = [Hx4yjax,Hx3y2jax]
                n = min(len(arg_vals),len(funcs))
                flat = len(arg_vals[0].shape) == 1
                dim0 = arg_vals[0].shape[0]
                if full:
                        dim1 = self.basisClass.numBasisFuncFull
                else:
                        dim1 = self.basisClass.numBasisFunc
                out_tans = np.zeros((dim0,dim1))
                for k in range(n):
                        if not (type(arg_tans[k]) is ad.Zero):
                                if type(arg_tans[k]) is batching.BatchTracer:
                                        flag = onp.any(arg_tans[k].val != 0)
                                else:
                                        flag = onp.any(arg_tans[k] != 0)
                                if flag:
                                        if flat:
                                                out_tans += funcs[k](*arg_vals,full=full)*np.expand_dims(arg_tans[k],1)
                                        else:
                                                out_tans += funcs[k](*arg_vals,full=full)*arg_tans[k]
                return (Hx3yjax(*arg_vals,full=full),out_tans)
        def Hx4_jvp(arg_vals,arg_tans,full=False):
                funcs = [Hx5jax,Hx4yjax]
                n = min(len(arg_vals),len(funcs))
                flat = len(arg_vals[0].shape) == 1
                dim0 = arg_vals[0].shape[0]
                if full:
                        dim1 = self.basisClass.numBasisFuncFull
                else:
                        dim1 = self.basisClass.numBasisFunc
                out_tans = np.zeros((dim0,dim1))
                for k in range(n):
                        if not (type(arg_tans[k]) is ad.Zero):
                                if type(arg_tans[k]) is batching.BatchTracer:
                                        flag = onp.any(arg_tans[k].val != 0)
                                else:
                                        flag = onp.any(arg_tans[k] != 0)
                                if flag:
                                        if flat:
                                                out_tans += funcs[k](*arg_vals,full=full)*np.expand_dims(arg_tans[k],1)
                                        else:
                                                out_tans += funcs[k](*arg_vals,full=full)*arg_tans[k]
                return (Hx4jax(*arg_vals,full=full),out_tans)
        def Hy5_jvp(arg_vals,arg_tans,full=False):
                funcs = [Hxy5jax,Hy6jax]
                n = min(len(arg_vals),len(funcs))
                flat = len(arg_vals[0].shape) == 1
                dim0 = arg_vals[0].shape[0]
                if full:
                        dim1 = self.basisClass.numBasisFuncFull
                else:
                        dim1 = self.basisClass.numBasisFunc
                out_tans = np.zeros((dim0,dim1))
                for k in range(n):
                        if not (type(arg_tans[k]) is ad.Zero):
                                if type(arg_tans[k]) is batching.BatchTracer:
                                        flag = onp.any(arg_tans[k].val != 0)
                                else:
                                        flag = onp.any(arg_tans[k] != 0)
                                if flag:
                                        if flat:
                                                out_tans += funcs[k](*arg_vals,full=full)*np.expand_dims(arg_tans[k],1)
                                        else:
                                                out_tans += funcs[k](*arg_vals,full=full)*arg_tans[k]
                return (Hy5jax(*arg_vals,full=full),out_tans)
        def Hx2y3_jvp(arg_vals,arg_tans,full=False):
                funcs = [Hx3y3jax,Hx2y4jax]
                n = min(len(arg_vals),len(funcs))
                flat = len(arg_vals[0].shape) == 1
                dim0 = arg_vals[0].shape[0]
                if full:
                        dim1 = self.basisClass.numBasisFuncFull
                else:
                        dim1 = self.basisClass.numBasisFunc
                out_tans = np.zeros((dim0,dim1))
                for k in range(n):
                        if not (type(arg_tans[k]) is ad.Zero):
                                if type(arg_tans[k]) is batching.BatchTracer:
                                        flag = onp.any(arg_tans[k].val != 0)
                                else:
                                        flag = onp.any(arg_tans[k] != 0)
                                if flag:
                                        if flat:
                                                out_tans += funcs[k](*arg_vals,full=full)*np.expand_dims(arg_tans[k],1)
                                        else:
                                                out_tans += funcs[k](*arg_vals,full=full)*arg_tans[k]
                return (Hx2y3jax(*arg_vals,full=full),out_tans)
        def Hx3y2_jvp(arg_vals,arg_tans,full=False):
                funcs = [Hx4y2jax,Hx3y3jax]
                n = min(len(arg_vals),len(funcs))
                flat = len(arg_vals[0].shape) == 1
                dim0 = arg_vals[0].shape[0]
                if full:
                        dim1 = self.basisClass.numBasisFuncFull
                else:
                        dim1 = self.basisClass.numBasisFunc
                out_tans = np.zeros((dim0,dim1))
                for k in range(n):
                        if not (type(arg_tans[k]) is ad.Zero):
                                if type(arg_tans[k]) is batching.BatchTracer:
                                        flag = onp.any(arg_tans[k].val != 0)
                                else:
                                        flag = onp.any(arg_tans[k] != 0)
                                if flag:
                                        if flat:
                                                out_tans += funcs[k](*arg_vals,full=full)*np.expand_dims(arg_tans[k],1)
                                        else:
                                                out_tans += funcs[k](*arg_vals,full=full)*arg_tans[k]
                return (Hx3y2jax(*arg_vals,full=full),out_tans)
        def Hx5_jvp(arg_vals,arg_tans,full=False):
                funcs = [Hx6jax,Hx5yjax]
                n = min(len(arg_vals),len(funcs))
                flat = len(arg_vals[0].shape) == 1
                dim0 = arg_vals[0].shape[0]
                if full:
                        dim1 = self.basisClass.numBasisFuncFull
                else:
                        dim1 = self.basisClass.numBasisFunc
                out_tans = np.zeros((dim0,dim1))
                for k in range(n):
                        if not (type(arg_tans[k]) is ad.Zero):
                                if type(arg_tans[k]) is batching.BatchTracer:
                                        flag = onp.any(arg_tans[k].val != 0)
                                else:
                                        flag = onp.any(arg_tans[k] != 0)
                                if flag:
                                        if flat:
                                                out_tans += funcs[k](*arg_vals,full=full)*np.expand_dims(arg_tans[k],1)
                                        else:
                                                out_tans += funcs[k](*arg_vals,full=full)*arg_tans[k]
                return (Hx5jax(*arg_vals,full=full),out_tans)
        def Hy6_jvp(arg_vals,arg_tans,full=False):
                funcs = [Hxy6jax,Hy7jax]
                n = min(len(arg_vals),len(funcs))
                flat = len(arg_vals[0].shape) == 1
                dim0 = arg_vals[0].shape[0]
                if full:
                        dim1 = self.basisClass.numBasisFuncFull
                else:
                        dim1 = self.basisClass.numBasisFunc
                out_tans = np.zeros((dim0,dim1))
                for k in range(n):
                        if not (type(arg_tans[k]) is ad.Zero):
                                if type(arg_tans[k]) is batching.BatchTracer:
                                        flag = onp.any(arg_tans[k].val != 0)
                                else:
                                        flag = onp.any(arg_tans[k] != 0)
                                if flag:
                                        if flat:
                                                out_tans += funcs[k](*arg_vals,full=full)*np.expand_dims(arg_tans[k],1)
                                        else:
                                                out_tans += funcs[k](*arg_vals,full=full)*arg_tans[k]
                return (Hy6jax(*arg_vals,full=full),out_tans)
        def Hxy5_jvp(arg_vals,arg_tans,full=False):
                funcs = [Hx2y5jax,Hxy6jax]
                n = min(len(arg_vals),len(funcs))
                flat = len(arg_vals[0].shape) == 1
                dim0 = arg_vals[0].shape[0]
                if full:
                        dim1 = self.basisClass.numBasisFuncFull
                else:
                        dim1 = self.basisClass.numBasisFunc
                out_tans = np.zeros((dim0,dim1))
                for k in range(n):
                        if not (type(arg_tans[k]) is ad.Zero):
                                if type(arg_tans[k]) is batching.BatchTracer:
                                        flag = onp.any(arg_tans[k].val != 0)
                                else:
                                        flag = onp.any(arg_tans[k] != 0)
                                if flag:
                                        if flat:
                                                out_tans += funcs[k](*arg_vals,full=full)*np.expand_dims(arg_tans[k],1)
                                        else:
                                                out_tans += funcs[k](*arg_vals,full=full)*arg_tans[k]
                return (Hxy5jax(*arg_vals,full=full),out_tans)
        def Hx2y4_jvp(arg_vals,arg_tans,full=False):
                funcs = [Hx3y4jax,Hx2y5jax]
                n = min(len(arg_vals),len(funcs))
                flat = len(arg_vals[0].shape) == 1
                dim0 = arg_vals[0].shape[0]
                if full:
                        dim1 = self.basisClass.numBasisFuncFull
                else:
                        dim1 = self.basisClass.numBasisFunc
                out_tans = np.zeros((dim0,dim1))
                for k in range(n):
                        if not (type(arg_tans[k]) is ad.Zero):
                                if type(arg_tans[k]) is batching.BatchTracer:
                                        flag = onp.any(arg_tans[k].val != 0)
                                else:
                                        flag = onp.any(arg_tans[k] != 0)
                                if flag:
                                        if flat:
                                                out_tans += funcs[k](*arg_vals,full=full)*np.expand_dims(arg_tans[k],1)
                                        else:
                                                out_tans += funcs[k](*arg_vals,full=full)*arg_tans[k]
                return (Hx2y4jax(*arg_vals,full=full),out_tans)
        def Hx3y3_jvp(arg_vals,arg_tans,full=False):
                funcs = [Hx4y3jax,Hx3y4jax]
                n = min(len(arg_vals),len(funcs))
                flat = len(arg_vals[0].shape) == 1
                dim0 = arg_vals[0].shape[0]
                if full:
                        dim1 = self.basisClass.numBasisFuncFull
                else:
                        dim1 = self.basisClass.numBasisFunc
                out_tans = np.zeros((dim0,dim1))
                for k in range(n):
                        if not (type(arg_tans[k]) is ad.Zero):
                                if type(arg_tans[k]) is batching.BatchTracer:
                                        flag = onp.any(arg_tans[k].val != 0)
                                else:
                                        flag = onp.any(arg_tans[k] != 0)
                                if flag:
                                        if flat:
                                                out_tans += funcs[k](*arg_vals,full=full)*np.expand_dims(arg_tans[k],1)
                                        else:
                                                out_tans += funcs[k](*arg_vals,full=full)*arg_tans[k]
                return (Hx3y3jax(*arg_vals,full=full),out_tans)
        def Hx5y_jvp(arg_vals,arg_tans,full=False):
                funcs = [Hx6yjax,Hx5y2jax]
                n = min(len(arg_vals),len(funcs))
                flat = len(arg_vals[0].shape) == 1
                dim0 = arg_vals[0].shape[0]
                if full:
                        dim1 = self.basisClass.numBasisFuncFull
                else:
                        dim1 = self.basisClass.numBasisFunc
                out_tans = np.zeros((dim0,dim1))
                for k in range(n):
                        if not (type(arg_tans[k]) is ad.Zero):
                                if type(arg_tans[k]) is batching.BatchTracer:
                                        flag = onp.any(arg_tans[k].val != 0)
                                else:
                                        flag = onp.any(arg_tans[k] != 0)
                                if flag:
                                        if flat:
                                                out_tans += funcs[k](*arg_vals,full=full)*np.expand_dims(arg_tans[k],1)
                                        else:
                                                out_tans += funcs[k](*arg_vals,full=full)*arg_tans[k]
                return (Hx5yjax(*arg_vals,full=full),out_tans)
        def Hx2y5_jvp(arg_vals,arg_tans,full=False):
                funcs = [Hx3y5jax,Hx2y6jax]
                n = min(len(arg_vals),len(funcs))
                flat = len(arg_vals[0].shape) == 1
                dim0 = arg_vals[0].shape[0]
                if full:
                        dim1 = self.basisClass.numBasisFuncFull
                else:
                        dim1 = self.basisClass.numBasisFunc
                out_tans = np.zeros((dim0,dim1))
                for k in range(n):
                        if not (type(arg_tans[k]) is ad.Zero):
                                if type(arg_tans[k]) is batching.BatchTracer:
                                        flag = onp.any(arg_tans[k].val != 0)
                                else:
                                        flag = onp.any(arg_tans[k] != 0)
                                if flag:
                                        if flat:
                                                out_tans += funcs[k](*arg_vals,full=full)*np.expand_dims(arg_tans[k],1)
                                        else:
                                                out_tans += funcs[k](*arg_vals,full=full)*arg_tans[k]
                return (Hx2y5jax(*arg_vals,full=full),out_tans)
        def Hx3y4_jvp(arg_vals,arg_tans,full=False):
                funcs = [Hx4y4jax,Hx3y5jax]
                n = min(len(arg_vals),len(funcs))
                flat = len(arg_vals[0].shape) == 1
                dim0 = arg_vals[0].shape[0]
                if full:
                        dim1 = self.basisClass.numBasisFuncFull
                else:
                        dim1 = self.basisClass.numBasisFunc
                out_tans = np.zeros((dim0,dim1))
                for k in range(n):
                        if not (type(arg_tans[k]) is ad.Zero):
                                if type(arg_tans[k]) is batching.BatchTracer:
                                        flag = onp.any(arg_tans[k].val != 0)
                                else:
                                        flag = onp.any(arg_tans[k] != 0)
                                if flag:
                                        if flat:
                                                out_tans += funcs[k](*arg_vals,full=full)*np.expand_dims(arg_tans[k],1)
                                        else:
                                                out_tans += funcs[k](*arg_vals,full=full)*arg_tans[k]
                return (Hx3y4jax(*arg_vals,full=full),out_tans)
        def Hx4y3_jvp(arg_vals,arg_tans,full=False):
                funcs = [Hx5y3jax]
                n = min(len(arg_vals),len(funcs))
                flat = len(arg_vals[0].shape) == 1
                dim0 = arg_vals[0].shape[0]
                if full:
                        dim1 = self.basisClass.numBasisFuncFull
                else:
                        dim1 = self.basisClass.numBasisFunc
                out_tans = np.zeros((dim0,dim1))
                for k in range(n):
                        if not (type(arg_tans[k]) is ad.Zero):
                                if type(arg_tans[k]) is batching.BatchTracer:
                                        flag = onp.any(arg_tans[k].val != 0)
                                else:
                                        flag = onp.any(arg_tans[k] != 0)
                                if flag:
                                        if flat:
                                                out_tans += funcs[k](*arg_vals,full=full)*np.expand_dims(arg_tans[k],1)
                                        else:
                                                out_tans += funcs[k](*arg_vals,full=full)*arg_tans[k]
                return (Hx4y3jax(*arg_vals,full=full),out_tans)
        def Hx5y2_jvp(arg_vals,arg_tans,full=False):
                funcs = [Hx6y2jax]
                n = min(len(arg_vals),len(funcs))
                flat = len(arg_vals[0].shape) == 1
                dim0 = arg_vals[0].shape[0]
                if full:
                        dim1 = self.basisClass.numBasisFuncFull
                else:
                        dim1 = self.basisClass.numBasisFunc
                out_tans = np.zeros((dim0,dim1))
                for k in range(n):
                        if not (type(arg_tans[k]) is ad.Zero):
                                if type(arg_tans[k]) is batching.BatchTracer:
                                        flag = onp.any(arg_tans[k].val != 0)
                                else:
                                        flag = onp.any(arg_tans[k] != 0)
                                if flag:
                                        if flat:
                                                out_tans += funcs[k](*arg_vals,full=full)*np.expand_dims(arg_tans[k],1)
                                        else:
                                                out_tans += funcs[k](*arg_vals,full=full)*arg_tans[k]
                return (Hx5y2jax(*arg_vals,full=full),out_tans)
        def Hx5y3_jvp(arg_vals,arg_tans,full=False):
                funcs = [Hx6y3jax]
                n = min(len(arg_vals),len(funcs))
                flat = len(arg_vals[0].shape) == 1
                dim0 = arg_vals[0].shape[0]
                if full:
                        dim1 = self.basisClass.numBasisFuncFull
                else:
                        dim1 = self.basisClass.numBasisFunc
                out_tans = np.zeros((dim0,dim1))
                for k in range(n):
                        if not (type(arg_tans[k]) is ad.Zero):
                                if type(arg_tans[k]) is batching.BatchTracer:
                                        flag = onp.any(arg_tans[k].val != 0)
                                else:
                                        flag = onp.any(arg_tans[k] != 0)
                                if flag:
                                        if flat:
                                                out_tans += funcs[k](*arg_vals,full=full)*np.expand_dims(arg_tans[k],1)
                                        else:
                                                out_tans += funcs[k](*arg_vals,full=full)*arg_tans[k]
                return (Hx5y3jax(*arg_vals,full=full),out_tans)
        def Hx4y4_jvp(arg_vals,arg_tans,full=False):
                funcs = [Hx5y4jax,Hx4y5jax]
                n = min(len(arg_vals),len(funcs))
                flat = len(arg_vals[0].shape) == 1
                dim0 = arg_vals[0].shape[0]
                if full:
                        dim1 = self.basisClass.numBasisFuncFull
                else:
                        dim1 = self.basisClass.numBasisFunc
                out_tans = np.zeros((dim0,dim1))
                for k in range(n):
                        if not (type(arg_tans[k]) is ad.Zero):
                                if type(arg_tans[k]) is batching.BatchTracer:
                                        flag = onp.any(arg_tans[k].val != 0)
                                else:
                                        flag = onp.any(arg_tans[k] != 0)
                                if flag:
                                        if flat:
                                                out_tans += funcs[k](*arg_vals,full=full)*np.expand_dims(arg_tans[k],1)
                                        else:
                                                out_tans += funcs[k](*arg_vals,full=full)*arg_tans[k]
                return (Hx4y4jax(*arg_vals,full=full),out_tans)
        ad.primitive_jvps[H_p] = H_jvp
        ad.primitive_jvps[Hx_p] = Hx_jvp
        ad.primitive_jvps[Hy_p] = Hy_jvp
        ad.primitive_jvps[Hz_p] = Hz_jvp
        ad.primitive_jvps[Hw_p] = Hw_jvp
        ad.primitive_jvps[Hxy_p] = Hxy_jvp
        ad.primitive_jvps[Hxz_p] = Hxz_jvp
        ad.primitive_jvps[Hyz_p] = Hyz_jvp
        ad.primitive_jvps[Hx2_p] = Hx2_jvp
        ad.primitive_jvps[Hy2_p] = Hy2_jvp
        ad.primitive_jvps[Hz2_p] = Hz2_jvp
        ad.primitive_jvps[Hy3_p] = Hy3_jvp
        ad.primitive_jvps[Hxy2_p] = Hxy2_jvp
        ad.primitive_jvps[Hx2y_p] = Hx2y_jvp
        ad.primitive_jvps[Hx3_p] = Hx3_jvp
        ad.primitive_jvps[Hy4_p] = Hy4_jvp
        ad.primitive_jvps[Hxy3_p] = Hxy3_jvp
        ad.primitive_jvps[Hx2y2_p] = Hx2y2_jvp
        ad.primitive_jvps[Hx3y_p] = Hx3y_jvp
        ad.primitive_jvps[Hx4_p] = Hx4_jvp
        ad.primitive_jvps[Hy5_p] = Hy5_jvp
        ad.primitive_jvps[Hx2y3_p] = Hx2y3_jvp
        ad.primitive_jvps[Hx3y2_p] = Hx3y2_jvp
        ad.primitive_jvps[Hx5_p] = Hx5_jvp
        ad.primitive_jvps[Hy6_p] = Hy6_jvp
        ad.primitive_jvps[Hxy5_p] = Hxy5_jvp
        ad.primitive_jvps[Hx2y4_p] = Hx2y4_jvp
        ad.primitive_jvps[Hx3y3_p] = Hx3y3_jvp
        ad.primitive_jvps[Hx5y_p] = Hx5y_jvp
        ad.primitive_jvps[Hx2y5_p] = Hx2y5_jvp
        ad.primitive_jvps[Hx3y4_p] = Hx3y4_jvp
        ad.primitive_jvps[Hx4y3_p] = Hx4y3_jvp
        ad.primitive_jvps[Hx5y2_p] = Hx5y2_jvp
        ad.primitive_jvps[Hx5y3_p] = Hx5y3_jvp
        ad.primitive_jvps[Hx4y4_p] = Hx4y4_jvp

        self._Hjax = Hjax
        self._Hxjax = Hxjax
        self._Hx2jax = Hx2jax
        self._Hy2jax = Hy2jax
        self._Hxy2jax = Hxy2jax
        self._Hyjax = Hyjax
        self._Hxyjax = Hxyjax
        self._Hzjax = Hzjax
