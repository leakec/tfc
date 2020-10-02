useValDefault = self.useValDefault

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
Hx2y_p = core.Primitive("Hx2y")
Hx2z_p = core.Primitive("Hx2z")
Hxy2_p = core.Primitive("Hxy2")
Hy2z_p = core.Primitive("Hy2z")
Hx3_p = core.Primitive("Hx3")
Hy3_p = core.Primitive("Hy3")
Hz3_p = core.Primitive("Hz3")
Hxy3_p = core.Primitive("Hxy3")
Hx3y_p = core.Primitive("Hx3y")
Hx2y2_p = core.Primitive("Hx2y2")
Hx4_p = core.Primitive("Hx4")
Hy4_p = core.Primitive("Hy4")
Hxy4_p = core.Primitive("Hxy4")
Hx4y_p = core.Primitive("Hx4y")
Hx3y2_p = core.Primitive("Hx3y2")
Hx2y3_p = core.Primitive("Hx2y3")
Hx5_p = core.Primitive("Hx5")
Hy5_p = core.Primitive("Hy5")

def Hjax(*x,full=False,useVal=useValDefault):
	return H_p.bind(*x,full=full,useVal=useVal)
def Hxjax(*x,full=False,useVal=useValDefault):
	return Hx_p.bind(*x,full=full,useVal=useVal)
def Hyjax(*x,full=False,useVal=useValDefault):
	return Hy_p.bind(*x,full=full,useVal=useVal)
def Hzjax(*x,full=False,useVal=useValDefault):
	return Hz_p.bind(*x,full=full,useVal=useVal)
def Hwjax(*x,full=False,useVal=useValDefault):
	return Hw_p.bind(*x,full=full,useVal=useVal)
def Hxyjax(*x,full=False,useVal=useValDefault):
	return Hxy_p.bind(*x,full=full,useVal=useVal)
def Hxzjax(*x,full=False,useVal=useValDefault):
	return Hxz_p.bind(*x,full=full,useVal=useVal)
def Hxwjax(*x,full=False,useVal=useValDefault):
	return Hxw_p.bind(*x,full=full,useVal=useVal)
def Hyzjax(*x,full=False,useVal=useValDefault):
	return Hyz_p.bind(*x,full=full,useVal=useVal)
def Hywjax(*x,full=False,useVal=useValDefault):
	return Hyw_p.bind(*x,full=full,useVal=useVal)
def Hzwjax(*x,full=False,useVal=useValDefault):
	return Hzw_p.bind(*x,full=full,useVal=useVal)
def Hx2jax(*x,full=False,useVal=useValDefault):
	return Hx2_p.bind(*x,full=full,useVal=useVal)
def Hy2jax(*x,full=False,useVal=useValDefault):
	return Hy2_p.bind(*x,full=full,useVal=useVal)
def Hz2jax(*x,full=False,useVal=useValDefault):
	return Hz2_p.bind(*x,full=full,useVal=useVal)
def Hw2jax(*x,full=False,useVal=useValDefault):
	return Hw2_p.bind(*x,full=full,useVal=useVal)
def Hx2yjax(*x,full=False,useVal=useValDefault):
	return Hx2y_p.bind(*x,full=full,useVal=useVal)
def Hx2zjax(*x,full=False,useVal=useValDefault):
	return Hx2z_p.bind(*x,full=full,useVal=useVal)
def Hxy2jax(*x,full=False,useVal=useValDefault):
	return Hxy2_p.bind(*x,full=full,useVal=useVal)
def Hy2zjax(*x,full=False,useVal=useValDefault):
	return Hy2z_p.bind(*x,full=full,useVal=useVal)
def Hx3jax(*x,full=False,useVal=useValDefault):
	return Hx3_p.bind(*x,full=full,useVal=useVal)
def Hy3jax(*x,full=False,useVal=useValDefault):
	return Hy3_p.bind(*x,full=full,useVal=useVal)
def Hz3jax(*x,full=False,useVal=useValDefault):
	return Hz3_p.bind(*x,full=full,useVal=useVal)
def Hxy3jax(*x,full=False,useVal=useValDefault):
	return Hxy3_p.bind(*x,full=full,useVal=useVal)
def Hx3yjax(*x,full=False,useVal=useValDefault):
	return Hx3y_p.bind(*x,full=full,useVal=useVal)
def Hx2y2jax(*x,full=False,useVal=useValDefault):
	return Hx2y2_p.bind(*x,full=full,useVal=useVal)
def Hx4jax(*x,full=False,useVal=useValDefault):
	return Hx4_p.bind(*x,full=full,useVal=useVal)
def Hy4jax(*x,full=False,useVal=useValDefault):
	return Hy4_p.bind(*x,full=full,useVal=useVal)
def Hxy4jax(*x,full=False,useVal=useValDefault):
	return Hxy4_p.bind(*x,full=full,useVal=useVal)
def Hx4yjax(*x,full=False,useVal=useValDefault):
	return Hx4y_p.bind(*x,full=full,useVal=useVal)
def Hx3y2jax(*x,full=False,useVal=useValDefault):
	return Hx3y2_p.bind(*x,full=full,useVal=useVal)
def Hx2y3jax(*x,full=False,useVal=useValDefault):
	return Hx2y3_p.bind(*x,full=full,useVal=useVal)
def Hx5jax(*x,full=False,useVal=useValDefault):
	return Hx5_p.bind(*x,full=full,useVal=useVal)
def Hy5jax(*x,full=False,useVal=useValDefault):
	return Hy5_p.bind(*x,full=full,useVal=useVal)

# Implicit translations
def H_impl(*x,full=False,useVal=useValDefault):
	return self.basisClass.H(np.array(x),np.array([0],dtype=np.int32),full,useVal)
def Hx_impl(*x,full=False,useVal=useValDefault):
	return self.basisClass.H(np.array(x),np.array([1],dtype=np.int32),full,useVal)
def Hy_impl(*x,full=False,useVal=useValDefault):
	return self.basisClass.H(np.array(x),np.array([0,1],dtype=np.int32),full,useVal)
def Hz_impl(*x,full=False,useVal=useValDefault):
	return self.basisClass.H(np.array(x),np.array([0,0,1],dtype=np.int32),full,useVal)
def Hw_impl(*x,full=False,useVal=useValDefault):
	return self.basisClass.H(np.array(x),np.array([0,0,0,1],dtype=np.int32),full,useVal)
def Hxy_impl(*x,full=False,useVal=useValDefault):
	return self.basisClass.H(np.array(x),np.array([1,1],dtype=np.int32),full,useVal)
def Hxz_impl(*x,full=False,useVal=useValDefault):
	return self.basisClass.H(np.array(x),np.array([1,0,1],dtype=np.int32),full,useVal)
def Hxw_impl(*x,full=False,useVal=useValDefault):
	return self.basisClass.H(np.array(x),np.array([1,0,0,1],dtype=np.int32),full,useVal)
def Hyz_impl(*x,full=False,useVal=useValDefault):
	return self.basisClass.H(np.array(x),np.array([0,1,1],dtype=np.int32),full,useVal)
def Hyw_impl(*x,full=False,useVal=useValDefault):
	return self.basisClass.H(np.array(x),np.array([0,1,0,1],dtype=np.int32),full,useVal)
def Hzw_impl(*x,full=False,useVal=useValDefault):
	return self.basisClass.H(np.array(x),np.array([0,0,1,1],dtype=np.int32),full,useVal)
def Hx2_impl(*x,full=False,useVal=useValDefault):
	return self.basisClass.H(np.array(x),np.array([2],dtype=np.int32),full,useVal)
def Hy2_impl(*x,full=False,useVal=useValDefault):
	return self.basisClass.H(np.array(x),np.array([0,2],dtype=np.int32),full,useVal)
def Hz2_impl(*x,full=False,useVal=useValDefault):
	return self.basisClass.H(np.array(x),np.array([0,0,2],dtype=np.int32),full,useVal)
def Hw2_impl(*x,full=False,useVal=useValDefault):
	return self.basisClass.H(np.array(x),np.array([0,0,0,2],dtype=np.int32),full,useVal)
def Hx2y_impl(*x,full=False,useVal=useValDefault):
	return self.basisClass.H(np.array(x),np.array([2,1],dtype=np.int32),full,useVal)
def Hx2z_impl(*x,full=False,useVal=useValDefault):
	return self.basisClass.H(np.array(x),np.array([2,0,1],dtype=np.int32),full,useVal)
def Hxy2_impl(*x,full=False,useVal=useValDefault):
	return self.basisClass.H(np.array(x),np.array([1,2],dtype=np.int32),full,useVal)
def Hy2z_impl(*x,full=False,useVal=useValDefault):
	return self.basisClass.H(np.array(x),np.array([0,2,1],dtype=np.int32),full,useVal)
def Hx3_impl(*x,full=False,useVal=useValDefault):
	return self.basisClass.H(np.array(x),np.array([3],dtype=np.int32),full,useVal)
def Hy3_impl(*x,full=False,useVal=useValDefault):
	return self.basisClass.H(np.array(x),np.array([0,3],dtype=np.int32),full,useVal)
def Hz3_impl(*x,full=False,useVal=useValDefault):
	return self.basisClass.H(np.array(x),np.array([0,0,3],dtype=np.int32),full,useVal)
def Hxy3_impl(*x,full=False,useVal=useValDefault):
	return self.basisClass.H(np.array(x),np.array([1,3],dtype=np.int32),full,useVal)
def Hx3y_impl(*x,full=False,useVal=useValDefault):
	return self.basisClass.H(np.array(x),np.array([3,1],dtype=np.int32),full,useVal)
def Hx2y2_impl(*x,full=False,useVal=useValDefault):
	return self.basisClass.H(np.array(x),np.array([2,2],dtype=np.int32),full,useVal)
def Hx4_impl(*x,full=False,useVal=useValDefault):
	return self.basisClass.H(np.array(x),np.array([4],dtype=np.int32),full,useVal)
def Hy4_impl(*x,full=False,useVal=useValDefault):
	return self.basisClass.H(np.array(x),np.array([0,4],dtype=np.int32),full,useVal)
def Hxy4_impl(*x,full=False,useVal=useValDefault):
	return self.basisClass.H(np.array(x),np.array([1,4],dtype=np.int32),full,useVal)
def Hx4y_impl(*x,full=False,useVal=useValDefault):
	return self.basisClass.H(np.array(x),np.array([4,1],dtype=np.int32),full,useVal)
def Hx3y2_impl(*x,full=False,useVal=useValDefault):
	return self.basisClass.H(np.array(x),np.array([3,2],dtype=np.int32),full,useVal)
def Hx2y3_impl(*x,full=False,useVal=useValDefault):
	return self.basisClass.H(np.array(x),np.array([2,3],dtype=np.int32),full,useVal)
def Hx5_impl(*x,full=False,useVal=useValDefault):
	return self.basisClass.H(np.array(x),np.array([5],dtype=np.int32),full,useVal)
def Hy5_impl(*x,full=False,useVal=useValDefault):
	return self.basisClass.H(np.array(x),np.array([0,5],dtype=np.int32),full,useVal)

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
Hx2y_p.def_impl(Hx2y_impl)
Hx2z_p.def_impl(Hx2z_impl)
Hxy2_p.def_impl(Hxy2_impl)
Hy2z_p.def_impl(Hy2z_impl)
Hx3_p.def_impl(Hx3_impl)
Hy3_p.def_impl(Hy3_impl)
Hz3_p.def_impl(Hz3_impl)
Hxy3_p.def_impl(Hxy3_impl)
Hx3y_p.def_impl(Hx3y_impl)
Hx2y2_p.def_impl(Hx2y2_impl)
Hx4_p.def_impl(Hx4_impl)
Hy4_p.def_impl(Hy4_impl)
Hxy4_p.def_impl(Hxy4_impl)
Hx4y_p.def_impl(Hx4y_impl)
Hx3y2_p.def_impl(Hx3y2_impl)
Hx2y3_p.def_impl(Hx2y3_impl)
Hx5_p.def_impl(Hx5_impl)
Hy5_p.def_impl(Hy5_impl)

def H_abstract_eval(*x,full=False,useVal=useValDefault):
	if any(useVal):
		dim0 = x[0].shape[0]
	else:
		dim0 = self.basisClass.n
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
Hx2y_p.def_abstract_eval(H_abstract_eval)
Hx2z_p.def_abstract_eval(H_abstract_eval)
Hxy2_p.def_abstract_eval(H_abstract_eval)
Hy2z_p.def_abstract_eval(H_abstract_eval)
Hx3_p.def_abstract_eval(H_abstract_eval)
Hy3_p.def_abstract_eval(H_abstract_eval)
Hz3_p.def_abstract_eval(H_abstract_eval)
Hxy3_p.def_abstract_eval(H_abstract_eval)
Hx3y_p.def_abstract_eval(H_abstract_eval)
Hx2y2_p.def_abstract_eval(H_abstract_eval)
Hx4_p.def_abstract_eval(H_abstract_eval)
Hy4_p.def_abstract_eval(H_abstract_eval)
Hxy4_p.def_abstract_eval(H_abstract_eval)
Hx4y_p.def_abstract_eval(H_abstract_eval)
Hx3y2_p.def_abstract_eval(H_abstract_eval)
Hx2y3_p.def_abstract_eval(H_abstract_eval)
Hx5_p.def_abstract_eval(H_abstract_eval)
Hy5_p.def_abstract_eval(H_abstract_eval)

# XLA compilation
def H_xla(c,*x,full=False,useVal=useValDefault):
	c = _unpack_builder(c)
	x_shape = c.get_shape(x[0])
	dims = x_shape.dimensions()
	dtype = x_shape.element_type()
	if any(useVal):
		dim0 = dims[0]
	else:
		dim0 = self.basisClass.n
	if full:
		dim1 = self.basisClass.numBasisFuncFull
	else:
		dim1 = self.basisClass.numBasisFunc
	return xla_client.ops.CustomCall(c, xlaName, (_constant_s32_scalar(c,self.basisClass.identifier),
                                                      xla_client.ops.ConcatInDim(c,x,0),
                                                      _constant_array(c,np.array([0],dtype=np.int32)),
                                                      _constant_s32_scalar(c,1),
                                                      _constant_bool(c,full),
                                                      _constant_array(c,useVal),
                                                      _constant_s32_scalar(c,dim0),
                                                      _constant_s32_scalar(c,dim1)
                                                 ),
                                                 xla_client.Shape.array_shape(dtype,(dim0,dim1)))
def Hx_xla(c,*x,full=False,useVal=useValDefault):
	c = _unpack_builder(c)
	x_shape = c.get_shape(x[0])
	dims = x_shape.dimensions()
	dtype = x_shape.element_type()
	if any(useVal):
		dim0 = dims[0]
	else:
		dim0 = self.basisClass.n
	if full:
		dim1 = self.basisClass.numBasisFuncFull
	else:
		dim1 = self.basisClass.numBasisFunc
	return xla_client.ops.CustomCall(c, xlaName, (_constant_s32_scalar(c,self.basisClass.identifier),
                                                      xla_client.ops.ConcatInDim(c,x,0),
                                                      _constant_array(c,np.array([1],dtype=np.int32)),
                                                      _constant_s32_scalar(c,1),
                                                      _constant_bool(c,full),
                                                      _constant_array(c,useVal),
                                                      _constant_s32_scalar(c,dim0),
                                                      _constant_s32_scalar(c,dim1)
                                                 ),
                                                 xla_client.Shape.array_shape(dtype,(dim0,dim1)))
def Hy_xla(c,*x,full=False,useVal=useValDefault):
	c = _unpack_builder(c)
	x_shape = c.get_shape(x[0])
	dims = x_shape.dimensions()
	dtype = x_shape.element_type()
	if any(useVal):
		dim0 = dims[0]
	else:
		dim0 = self.basisClass.n
	if full:
		dim1 = self.basisClass.numBasisFuncFull
	else:
		dim1 = self.basisClass.numBasisFunc
	return xla_client.ops.CustomCall(c, xlaName, (_constant_s32_scalar(c,self.basisClass.identifier),
                                                      xla_client.ops.ConcatInDim(c,x,0),
                                                      _constant_array(c,np.array([0,1],dtype=np.int32)),
                                                      _constant_s32_scalar(c,2),
                                                      _constant_bool(c,full),
                                                      _constant_array(c,useVal),
                                                      _constant_s32_scalar(c,dim0),
                                                      _constant_s32_scalar(c,dim1)
                                                 ),
                                                 xla_client.Shape.array_shape(dtype,(dim0,dim1)))
def Hz_xla(c,*x,full=False,useVal=useValDefault):
	c = _unpack_builder(c)
	x_shape = c.get_shape(x[0])
	dims = x_shape.dimensions()
	dtype = x_shape.element_type()
	if any(useVal):
		dim0 = dims[0]
	else:
		dim0 = self.basisClass.n
	if full:
		dim1 = self.basisClass.numBasisFuncFull
	else:
		dim1 = self.basisClass.numBasisFunc
	return xla_client.ops.CustomCall(c, xlaName, (_constant_s32_scalar(c,self.basisClass.identifier),
                                                      xla_client.ops.ConcatInDim(c,x,0),
                                                      _constant_array(c,np.array([0,0,1],dtype=np.int32)),
                                                      _constant_s32_scalar(c,3),
                                                      _constant_bool(c,full),
                                                      _constant_array(c,useVal),
                                                      _constant_s32_scalar(c,dim0),
                                                      _constant_s32_scalar(c,dim1)
                                                 ),
                                                 xla_client.Shape.array_shape(dtype,(dim0,dim1)))
def Hw_xla(c,*x,full=False,useVal=useValDefault):
	c = _unpack_builder(c)
	x_shape = c.get_shape(x[0])
	dims = x_shape.dimensions()
	dtype = x_shape.element_type()
	if any(useVal):
		dim0 = dims[0]
	else:
		dim0 = self.basisClass.n
	if full:
		dim1 = self.basisClass.numBasisFuncFull
	else:
		dim1 = self.basisClass.numBasisFunc
	return xla_client.ops.CustomCall(c, xlaName, (_constant_s32_scalar(c,self.basisClass.identifier),
                                                      xla_client.ops.ConcatInDim(c,x,0),
                                                      _constant_array(c,np.array([0,0,0,1],dtype=np.int32)),
                                                      _constant_s32_scalar(c,4),
                                                      _constant_bool(c,full),
                                                      _constant_array(c,useVal),
                                                      _constant_s32_scalar(c,dim0),
                                                      _constant_s32_scalar(c,dim1)
                                                 ),
                                                 xla_client.Shape.array_shape(dtype,(dim0,dim1)))
def Hxy_xla(c,*x,full=False,useVal=useValDefault):
	c = _unpack_builder(c)
	x_shape = c.get_shape(x[0])
	dims = x_shape.dimensions()
	dtype = x_shape.element_type()
	if any(useVal):
		dim0 = dims[0]
	else:
		dim0 = self.basisClass.n
	if full:
		dim1 = self.basisClass.numBasisFuncFull
	else:
		dim1 = self.basisClass.numBasisFunc
	return xla_client.ops.CustomCall(c, xlaName, (_constant_s32_scalar(c,self.basisClass.identifier),
                                                      xla_client.ops.ConcatInDim(c,x,0),
                                                      _constant_array(c,np.array([1,1],dtype=np.int32)),
                                                      _constant_s32_scalar(c,2),
                                                      _constant_bool(c,full),
                                                      _constant_array(c,useVal),
                                                      _constant_s32_scalar(c,dim0),
                                                      _constant_s32_scalar(c,dim1)
                                                 ),
                                                 xla_client.Shape.array_shape(dtype,(dim0,dim1)))
def Hxz_xla(c,*x,full=False,useVal=useValDefault):
	c = _unpack_builder(c)
	x_shape = c.get_shape(x[0])
	dims = x_shape.dimensions()
	dtype = x_shape.element_type()
	if any(useVal):
		dim0 = dims[0]
	else:
		dim0 = self.basisClass.n
	if full:
		dim1 = self.basisClass.numBasisFuncFull
	else:
		dim1 = self.basisClass.numBasisFunc
	return xla_client.ops.CustomCall(c, xlaName, (_constant_s32_scalar(c,self.basisClass.identifier),
                                                      xla_client.ops.ConcatInDim(c,x,0),
                                                      _constant_array(c,np.array([1,0,1],dtype=np.int32)),
                                                      _constant_s32_scalar(c,3),
                                                      _constant_bool(c,full),
                                                      _constant_array(c,useVal),
                                                      _constant_s32_scalar(c,dim0),
                                                      _constant_s32_scalar(c,dim1)
                                                 ),
                                                 xla_client.Shape.array_shape(dtype,(dim0,dim1)))
def Hxw_xla(c,*x,full=False,useVal=useValDefault):
	c = _unpack_builder(c)
	x_shape = c.get_shape(x[0])
	dims = x_shape.dimensions()
	dtype = x_shape.element_type()
	if any(useVal):
		dim0 = dims[0]
	else:
		dim0 = self.basisClass.n
	if full:
		dim1 = self.basisClass.numBasisFuncFull
	else:
		dim1 = self.basisClass.numBasisFunc
	return xla_client.ops.CustomCall(c, xlaName, (_constant_s32_scalar(c,self.basisClass.identifier),
                                                      xla_client.ops.ConcatInDim(c,x,0),
                                                      _constant_array(c,np.array([1,0,0,1],dtype=np.int32)),
                                                      _constant_s32_scalar(c,4),
                                                      _constant_bool(c,full),
                                                      _constant_array(c,useVal),
                                                      _constant_s32_scalar(c,dim0),
                                                      _constant_s32_scalar(c,dim1)
                                                 ),
                                                 xla_client.Shape.array_shape(dtype,(dim0,dim1)))
def Hyz_xla(c,*x,full=False,useVal=useValDefault):
	c = _unpack_builder(c)
	x_shape = c.get_shape(x[0])
	dims = x_shape.dimensions()
	dtype = x_shape.element_type()
	if any(useVal):
		dim0 = dims[0]
	else:
		dim0 = self.basisClass.n
	if full:
		dim1 = self.basisClass.numBasisFuncFull
	else:
		dim1 = self.basisClass.numBasisFunc
	return xla_client.ops.CustomCall(c, xlaName, (_constant_s32_scalar(c,self.basisClass.identifier),
                                                      xla_client.ops.ConcatInDim(c,x,0),
                                                      _constant_array(c,np.array([0,1,1],dtype=np.int32)),
                                                      _constant_s32_scalar(c,3),
                                                      _constant_bool(c,full),
                                                      _constant_array(c,useVal),
                                                      _constant_s32_scalar(c,dim0),
                                                      _constant_s32_scalar(c,dim1)
                                                 ),
                                                 xla_client.Shape.array_shape(dtype,(dim0,dim1)))
def Hyw_xla(c,*x,full=False,useVal=useValDefault):
	c = _unpack_builder(c)
	x_shape = c.get_shape(x[0])
	dims = x_shape.dimensions()
	dtype = x_shape.element_type()
	if any(useVal):
		dim0 = dims[0]
	else:
		dim0 = self.basisClass.n
	if full:
		dim1 = self.basisClass.numBasisFuncFull
	else:
		dim1 = self.basisClass.numBasisFunc
	return xla_client.ops.CustomCall(c, xlaName, (_constant_s32_scalar(c,self.basisClass.identifier),
                                                      xla_client.ops.ConcatInDim(c,x,0),
                                                      _constant_array(c,np.array([0,1,0,1],dtype=np.int32)),
                                                      _constant_s32_scalar(c,4),
                                                      _constant_bool(c,full),
                                                      _constant_array(c,useVal),
                                                      _constant_s32_scalar(c,dim0),
                                                      _constant_s32_scalar(c,dim1)
                                                 ),
                                                 xla_client.Shape.array_shape(dtype,(dim0,dim1)))
def Hzw_xla(c,*x,full=False,useVal=useValDefault):
	c = _unpack_builder(c)
	x_shape = c.get_shape(x[0])
	dims = x_shape.dimensions()
	dtype = x_shape.element_type()
	if any(useVal):
		dim0 = dims[0]
	else:
		dim0 = self.basisClass.n
	if full:
		dim1 = self.basisClass.numBasisFuncFull
	else:
		dim1 = self.basisClass.numBasisFunc
	return xla_client.ops.CustomCall(c, xlaName, (_constant_s32_scalar(c,self.basisClass.identifier),
                                                      xla_client.ops.ConcatInDim(c,x,0),
                                                      _constant_array(c,np.array([0,0,1,1],dtype=np.int32)),
                                                      _constant_s32_scalar(c,4),
                                                      _constant_bool(c,full),
                                                      _constant_array(c,useVal),
                                                      _constant_s32_scalar(c,dim0),
                                                      _constant_s32_scalar(c,dim1)
                                                 ),
                                                 xla_client.Shape.array_shape(dtype,(dim0,dim1)))
def Hx2_xla(c,*x,full=False,useVal=useValDefault):
	c = _unpack_builder(c)
	x_shape = c.get_shape(x[0])
	dims = x_shape.dimensions()
	dtype = x_shape.element_type()
	if any(useVal):
		dim0 = dims[0]
	else:
		dim0 = self.basisClass.n
	if full:
		dim1 = self.basisClass.numBasisFuncFull
	else:
		dim1 = self.basisClass.numBasisFunc
	return xla_client.ops.CustomCall(c, xlaName, (_constant_s32_scalar(c,self.basisClass.identifier),
                                                      xla_client.ops.ConcatInDim(c,x,0),
                                                      _constant_array(c,np.array([2],dtype=np.int32)),
                                                      _constant_s32_scalar(c,1),
                                                      _constant_bool(c,full),
                                                      _constant_array(c,useVal),
                                                      _constant_s32_scalar(c,dim0),
                                                      _constant_s32_scalar(c,dim1)
                                                 ),
                                                 xla_client.Shape.array_shape(dtype,(dim0,dim1)))
def Hy2_xla(c,*x,full=False,useVal=useValDefault):
	c = _unpack_builder(c)
	x_shape = c.get_shape(x[0])
	dims = x_shape.dimensions()
	dtype = x_shape.element_type()
	if any(useVal):
		dim0 = dims[0]
	else:
		dim0 = self.basisClass.n
	if full:
		dim1 = self.basisClass.numBasisFuncFull
	else:
		dim1 = self.basisClass.numBasisFunc
	return xla_client.ops.CustomCall(c, xlaName, (_constant_s32_scalar(c,self.basisClass.identifier),
                                                      xla_client.ops.ConcatInDim(c,x,0),
                                                      _constant_array(c,np.array([0,2],dtype=np.int32)),
                                                      _constant_s32_scalar(c,2),
                                                      _constant_bool(c,full),
                                                      _constant_array(c,useVal),
                                                      _constant_s32_scalar(c,dim0),
                                                      _constant_s32_scalar(c,dim1)
                                                 ),
                                                 xla_client.Shape.array_shape(dtype,(dim0,dim1)))
def Hz2_xla(c,*x,full=False,useVal=useValDefault):
	c = _unpack_builder(c)
	x_shape = c.get_shape(x[0])
	dims = x_shape.dimensions()
	dtype = x_shape.element_type()
	if any(useVal):
		dim0 = dims[0]
	else:
		dim0 = self.basisClass.n
	if full:
		dim1 = self.basisClass.numBasisFuncFull
	else:
		dim1 = self.basisClass.numBasisFunc
	return xla_client.ops.CustomCall(c, xlaName, (_constant_s32_scalar(c,self.basisClass.identifier),
                                                      xla_client.ops.ConcatInDim(c,x,0),
                                                      _constant_array(c,np.array([0,0,2],dtype=np.int32)),
                                                      _constant_s32_scalar(c,3),
                                                      _constant_bool(c,full),
                                                      _constant_array(c,useVal),
                                                      _constant_s32_scalar(c,dim0),
                                                      _constant_s32_scalar(c,dim1)
                                                 ),
                                                 xla_client.Shape.array_shape(dtype,(dim0,dim1)))
def Hw2_xla(c,*x,full=False,useVal=useValDefault):
	c = _unpack_builder(c)
	x_shape = c.get_shape(x[0])
	dims = x_shape.dimensions()
	dtype = x_shape.element_type()
	if any(useVal):
		dim0 = dims[0]
	else:
		dim0 = self.basisClass.n
	if full:
		dim1 = self.basisClass.numBasisFuncFull
	else:
		dim1 = self.basisClass.numBasisFunc
	return xla_client.ops.CustomCall(c, xlaName, (_constant_s32_scalar(c,self.basisClass.identifier),
                                                      xla_client.ops.ConcatInDim(c,x,0),
                                                      _constant_array(c,np.array([0,0,0,2],dtype=np.int32)),
                                                      _constant_s32_scalar(c,4),
                                                      _constant_bool(c,full),
                                                      _constant_array(c,useVal),
                                                      _constant_s32_scalar(c,dim0),
                                                      _constant_s32_scalar(c,dim1)
                                                 ),
                                                 xla_client.Shape.array_shape(dtype,(dim0,dim1)))
def Hx2y_xla(c,*x,full=False,useVal=useValDefault):
	c = _unpack_builder(c)
	x_shape = c.get_shape(x[0])
	dims = x_shape.dimensions()
	dtype = x_shape.element_type()
	if any(useVal):
		dim0 = dims[0]
	else:
		dim0 = self.basisClass.n
	if full:
		dim1 = self.basisClass.numBasisFuncFull
	else:
		dim1 = self.basisClass.numBasisFunc
	return xla_client.ops.CustomCall(c, xlaName, (_constant_s32_scalar(c,self.basisClass.identifier),
                                                      xla_client.ops.ConcatInDim(c,x,0),
                                                      _constant_array(c,np.array([2,1],dtype=np.int32)),
                                                      _constant_s32_scalar(c,2),
                                                      _constant_bool(c,full),
                                                      _constant_array(c,useVal),
                                                      _constant_s32_scalar(c,dim0),
                                                      _constant_s32_scalar(c,dim1)
                                                 ),
                                                 xla_client.Shape.array_shape(dtype,(dim0,dim1)))
def Hx2z_xla(c,*x,full=False,useVal=useValDefault):
	c = _unpack_builder(c)
	x_shape = c.get_shape(x[0])
	dims = x_shape.dimensions()
	dtype = x_shape.element_type()
	if any(useVal):
		dim0 = dims[0]
	else:
		dim0 = self.basisClass.n
	if full:
		dim1 = self.basisClass.numBasisFuncFull
	else:
		dim1 = self.basisClass.numBasisFunc
	return xla_client.ops.CustomCall(c, xlaName, (_constant_s32_scalar(c,self.basisClass.identifier),
                                                      xla_client.ops.ConcatInDim(c,x,0),
                                                      _constant_array(c,np.array([2,0,1],dtype=np.int32)),
                                                      _constant_s32_scalar(c,3),
                                                      _constant_bool(c,full),
                                                      _constant_array(c,useVal),
                                                      _constant_s32_scalar(c,dim0),
                                                      _constant_s32_scalar(c,dim1)
                                                 ),
                                                 xla_client.Shape.array_shape(dtype,(dim0,dim1)))
def Hxy2_xla(c,*x,full=False,useVal=useValDefault):
	c = _unpack_builder(c)
	x_shape = c.get_shape(x[0])
	dims = x_shape.dimensions()
	dtype = x_shape.element_type()
	if any(useVal):
		dim0 = dims[0]
	else:
		dim0 = self.basisClass.n
	if full:
		dim1 = self.basisClass.numBasisFuncFull
	else:
		dim1 = self.basisClass.numBasisFunc
	return xla_client.ops.CustomCall(c, xlaName, (_constant_s32_scalar(c,self.basisClass.identifier),
                                                      xla_client.ops.ConcatInDim(c,x,0),
                                                      _constant_array(c,np.array([1,2],dtype=np.int32)),
                                                      _constant_s32_scalar(c,2),
                                                      _constant_bool(c,full),
                                                      _constant_array(c,useVal),
                                                      _constant_s32_scalar(c,dim0),
                                                      _constant_s32_scalar(c,dim1)
                                                 ),
                                                 xla_client.Shape.array_shape(dtype,(dim0,dim1)))
def Hy2z_xla(c,*x,full=False,useVal=useValDefault):
	c = _unpack_builder(c)
	x_shape = c.get_shape(x[0])
	dims = x_shape.dimensions()
	dtype = x_shape.element_type()
	if any(useVal):
		dim0 = dims[0]
	else:
		dim0 = self.basisClass.n
	if full:
		dim1 = self.basisClass.numBasisFuncFull
	else:
		dim1 = self.basisClass.numBasisFunc
	return xla_client.ops.CustomCall(c, xlaName, (_constant_s32_scalar(c,self.basisClass.identifier),
                                                      xla_client.ops.ConcatInDim(c,x,0),
                                                      _constant_array(c,np.array([0,2,1],dtype=np.int32)),
                                                      _constant_s32_scalar(c,3),
                                                      _constant_bool(c,full),
                                                      _constant_array(c,useVal),
                                                      _constant_s32_scalar(c,dim0),
                                                      _constant_s32_scalar(c,dim1)
                                                 ),
                                                 xla_client.Shape.array_shape(dtype,(dim0,dim1)))
def Hx3_xla(c,*x,full=False,useVal=useValDefault):
	c = _unpack_builder(c)
	x_shape = c.get_shape(x[0])
	dims = x_shape.dimensions()
	dtype = x_shape.element_type()
	if any(useVal):
		dim0 = dims[0]
	else:
		dim0 = self.basisClass.n
	if full:
		dim1 = self.basisClass.numBasisFuncFull
	else:
		dim1 = self.basisClass.numBasisFunc
	return xla_client.ops.CustomCall(c, xlaName, (_constant_s32_scalar(c,self.basisClass.identifier),
                                                      xla_client.ops.ConcatInDim(c,x,0),
                                                      _constant_array(c,np.array([3],dtype=np.int32)),
                                                      _constant_s32_scalar(c,1),
                                                      _constant_bool(c,full),
                                                      _constant_array(c,useVal),
                                                      _constant_s32_scalar(c,dim0),
                                                      _constant_s32_scalar(c,dim1)
                                                 ),
                                                 xla_client.Shape.array_shape(dtype,(dim0,dim1)))
def Hy3_xla(c,*x,full=False,useVal=useValDefault):
	c = _unpack_builder(c)
	x_shape = c.get_shape(x[0])
	dims = x_shape.dimensions()
	dtype = x_shape.element_type()
	if any(useVal):
		dim0 = dims[0]
	else:
		dim0 = self.basisClass.n
	if full:
		dim1 = self.basisClass.numBasisFuncFull
	else:
		dim1 = self.basisClass.numBasisFunc
	return xla_client.ops.CustomCall(c, xlaName, (_constant_s32_scalar(c,self.basisClass.identifier),
                                                      xla_client.ops.ConcatInDim(c,x,0),
                                                      _constant_array(c,np.array([0,3],dtype=np.int32)),
                                                      _constant_s32_scalar(c,2),
                                                      _constant_bool(c,full),
                                                      _constant_array(c,useVal),
                                                      _constant_s32_scalar(c,dim0),
                                                      _constant_s32_scalar(c,dim1)
                                                 ),
                                                 xla_client.Shape.array_shape(dtype,(dim0,dim1)))
def Hz3_xla(c,*x,full=False,useVal=useValDefault):
	c = _unpack_builder(c)
	x_shape = c.get_shape(x[0])
	dims = x_shape.dimensions()
	dtype = x_shape.element_type()
	if any(useVal):
		dim0 = dims[0]
	else:
		dim0 = self.basisClass.n
	if full:
		dim1 = self.basisClass.numBasisFuncFull
	else:
		dim1 = self.basisClass.numBasisFunc
	return xla_client.ops.CustomCall(c, xlaName, (_constant_s32_scalar(c,self.basisClass.identifier),
                                                      xla_client.ops.ConcatInDim(c,x,0),
                                                      _constant_array(c,np.array([0,0,3],dtype=np.int32)),
                                                      _constant_s32_scalar(c,3),
                                                      _constant_bool(c,full),
                                                      _constant_array(c,useVal),
                                                      _constant_s32_scalar(c,dim0),
                                                      _constant_s32_scalar(c,dim1)
                                                 ),
                                                 xla_client.Shape.array_shape(dtype,(dim0,dim1)))
def Hxy3_xla(c,*x,full=False,useVal=useValDefault):
	c = _unpack_builder(c)
	x_shape = c.get_shape(x[0])
	dims = x_shape.dimensions()
	dtype = x_shape.element_type()
	if any(useVal):
		dim0 = dims[0]
	else:
		dim0 = self.basisClass.n
	if full:
		dim1 = self.basisClass.numBasisFuncFull
	else:
		dim1 = self.basisClass.numBasisFunc
	return xla_client.ops.CustomCall(c, xlaName, (_constant_s32_scalar(c,self.basisClass.identifier),
                                                      xla_client.ops.ConcatInDim(c,x,0),
                                                      _constant_array(c,np.array([1,3],dtype=np.int32)),
                                                      _constant_s32_scalar(c,2),
                                                      _constant_bool(c,full),
                                                      _constant_array(c,useVal),
                                                      _constant_s32_scalar(c,dim0),
                                                      _constant_s32_scalar(c,dim1)
                                                 ),
                                                 xla_client.Shape.array_shape(dtype,(dim0,dim1)))
def Hx3y_xla(c,*x,full=False,useVal=useValDefault):
	c = _unpack_builder(c)
	x_shape = c.get_shape(x[0])
	dims = x_shape.dimensions()
	dtype = x_shape.element_type()
	if any(useVal):
		dim0 = dims[0]
	else:
		dim0 = self.basisClass.n
	if full:
		dim1 = self.basisClass.numBasisFuncFull
	else:
		dim1 = self.basisClass.numBasisFunc
	return xla_client.ops.CustomCall(c, xlaName, (_constant_s32_scalar(c,self.basisClass.identifier),
                                                      xla_client.ops.ConcatInDim(c,x,0),
                                                      _constant_array(c,np.array([3,1],dtype=np.int32)),
                                                      _constant_s32_scalar(c,2),
                                                      _constant_bool(c,full),
                                                      _constant_array(c,useVal),
                                                      _constant_s32_scalar(c,dim0),
                                                      _constant_s32_scalar(c,dim1)
                                                 ),
                                                 xla_client.Shape.array_shape(dtype,(dim0,dim1)))
def Hx2y2_xla(c,*x,full=False,useVal=useValDefault):
	c = _unpack_builder(c)
	x_shape = c.get_shape(x[0])
	dims = x_shape.dimensions()
	dtype = x_shape.element_type()
	if any(useVal):
		dim0 = dims[0]
	else:
		dim0 = self.basisClass.n
	if full:
		dim1 = self.basisClass.numBasisFuncFull
	else:
		dim1 = self.basisClass.numBasisFunc
	return xla_client.ops.CustomCall(c, xlaName, (_constant_s32_scalar(c,self.basisClass.identifier),
                                                      xla_client.ops.ConcatInDim(c,x,0),
                                                      _constant_array(c,np.array([2,2],dtype=np.int32)),
                                                      _constant_s32_scalar(c,2),
                                                      _constant_bool(c,full),
                                                      _constant_array(c,useVal),
                                                      _constant_s32_scalar(c,dim0),
                                                      _constant_s32_scalar(c,dim1)
                                                 ),
                                                 xla_client.Shape.array_shape(dtype,(dim0,dim1)))
def Hx4_xla(c,*x,full=False,useVal=useValDefault):
	c = _unpack_builder(c)
	x_shape = c.get_shape(x[0])
	dims = x_shape.dimensions()
	dtype = x_shape.element_type()
	if any(useVal):
		dim0 = dims[0]
	else:
		dim0 = self.basisClass.n
	if full:
		dim1 = self.basisClass.numBasisFuncFull
	else:
		dim1 = self.basisClass.numBasisFunc
	return xla_client.ops.CustomCall(c, xlaName, (_constant_s32_scalar(c,self.basisClass.identifier),
                                                      xla_client.ops.ConcatInDim(c,x,0),
                                                      _constant_array(c,np.array([4],dtype=np.int32)),
                                                      _constant_s32_scalar(c,1),
                                                      _constant_bool(c,full),
                                                      _constant_array(c,useVal),
                                                      _constant_s32_scalar(c,dim0),
                                                      _constant_s32_scalar(c,dim1)
                                                 ),
                                                 xla_client.Shape.array_shape(dtype,(dim0,dim1)))
def Hy4_xla(c,*x,full=False,useVal=useValDefault):
	c = _unpack_builder(c)
	x_shape = c.get_shape(x[0])
	dims = x_shape.dimensions()
	dtype = x_shape.element_type()
	if any(useVal):
		dim0 = dims[0]
	else:
		dim0 = self.basisClass.n
	if full:
		dim1 = self.basisClass.numBasisFuncFull
	else:
		dim1 = self.basisClass.numBasisFunc
	return xla_client.ops.CustomCall(c, xlaName, (_constant_s32_scalar(c,self.basisClass.identifier),
                                                      xla_client.ops.ConcatInDim(c,x,0),
                                                      _constant_array(c,np.array([0,4],dtype=np.int32)),
                                                      _constant_s32_scalar(c,2),
                                                      _constant_bool(c,full),
                                                      _constant_array(c,useVal),
                                                      _constant_s32_scalar(c,dim0),
                                                      _constant_s32_scalar(c,dim1)
                                                 ),
                                                 xla_client.Shape.array_shape(dtype,(dim0,dim1)))
def Hxy4_xla(c,*x,full=False,useVal=useValDefault):
	c = _unpack_builder(c)
	x_shape = c.get_shape(x[0])
	dims = x_shape.dimensions()
	dtype = x_shape.element_type()
	if any(useVal):
		dim0 = dims[0]
	else:
		dim0 = self.basisClass.n
	if full:
		dim1 = self.basisClass.numBasisFuncFull
	else:
		dim1 = self.basisClass.numBasisFunc
	return xla_client.ops.CustomCall(c, xlaName, (_constant_s32_scalar(c,self.basisClass.identifier),
                                                      xla_client.ops.ConcatInDim(c,x,0),
                                                      _constant_array(c,np.array([1,4],dtype=np.int32)),
                                                      _constant_s32_scalar(c,2),
                                                      _constant_bool(c,full),
                                                      _constant_array(c,useVal),
                                                      _constant_s32_scalar(c,dim0),
                                                      _constant_s32_scalar(c,dim1)
                                                 ),
                                                 xla_client.Shape.array_shape(dtype,(dim0,dim1)))
def Hx4y_xla(c,*x,full=False,useVal=useValDefault):
	c = _unpack_builder(c)
	x_shape = c.get_shape(x[0])
	dims = x_shape.dimensions()
	dtype = x_shape.element_type()
	if any(useVal):
		dim0 = dims[0]
	else:
		dim0 = self.basisClass.n
	if full:
		dim1 = self.basisClass.numBasisFuncFull
	else:
		dim1 = self.basisClass.numBasisFunc
	return xla_client.ops.CustomCall(c, xlaName, (_constant_s32_scalar(c,self.basisClass.identifier),
                                                      xla_client.ops.ConcatInDim(c,x,0),
                                                      _constant_array(c,np.array([4,1],dtype=np.int32)),
                                                      _constant_s32_scalar(c,2),
                                                      _constant_bool(c,full),
                                                      _constant_array(c,useVal),
                                                      _constant_s32_scalar(c,dim0),
                                                      _constant_s32_scalar(c,dim1)
                                                 ),
                                                 xla_client.Shape.array_shape(dtype,(dim0,dim1)))
def Hx3y2_xla(c,*x,full=False,useVal=useValDefault):
	c = _unpack_builder(c)
	x_shape = c.get_shape(x[0])
	dims = x_shape.dimensions()
	dtype = x_shape.element_type()
	if any(useVal):
		dim0 = dims[0]
	else:
		dim0 = self.basisClass.n
	if full:
		dim1 = self.basisClass.numBasisFuncFull
	else:
		dim1 = self.basisClass.numBasisFunc
	return xla_client.ops.CustomCall(c, xlaName, (_constant_s32_scalar(c,self.basisClass.identifier),
                                                      xla_client.ops.ConcatInDim(c,x,0),
                                                      _constant_array(c,np.array([3,2],dtype=np.int32)),
                                                      _constant_s32_scalar(c,2),
                                                      _constant_bool(c,full),
                                                      _constant_array(c,useVal),
                                                      _constant_s32_scalar(c,dim0),
                                                      _constant_s32_scalar(c,dim1)
                                                 ),
                                                 xla_client.Shape.array_shape(dtype,(dim0,dim1)))
def Hx2y3_xla(c,*x,full=False,useVal=useValDefault):
	c = _unpack_builder(c)
	x_shape = c.get_shape(x[0])
	dims = x_shape.dimensions()
	dtype = x_shape.element_type()
	if any(useVal):
		dim0 = dims[0]
	else:
		dim0 = self.basisClass.n
	if full:
		dim1 = self.basisClass.numBasisFuncFull
	else:
		dim1 = self.basisClass.numBasisFunc
	return xla_client.ops.CustomCall(c, xlaName, (_constant_s32_scalar(c,self.basisClass.identifier),
                                                      xla_client.ops.ConcatInDim(c,x,0),
                                                      _constant_array(c,np.array([2,3],dtype=np.int32)),
                                                      _constant_s32_scalar(c,2),
                                                      _constant_bool(c,full),
                                                      _constant_array(c,useVal),
                                                      _constant_s32_scalar(c,dim0),
                                                      _constant_s32_scalar(c,dim1)
                                                 ),
                                                 xla_client.Shape.array_shape(dtype,(dim0,dim1)))
def Hx5_xla(c,*x,full=False,useVal=useValDefault):
	c = _unpack_builder(c)
	x_shape = c.get_shape(x[0])
	dims = x_shape.dimensions()
	dtype = x_shape.element_type()
	if any(useVal):
		dim0 = dims[0]
	else:
		dim0 = self.basisClass.n
	if full:
		dim1 = self.basisClass.numBasisFuncFull
	else:
		dim1 = self.basisClass.numBasisFunc
	return xla_client.ops.CustomCall(c, xlaName, (_constant_s32_scalar(c,self.basisClass.identifier),
                                                      xla_client.ops.ConcatInDim(c,x,0),
                                                      _constant_array(c,np.array([5],dtype=np.int32)),
                                                      _constant_s32_scalar(c,1),
                                                      _constant_bool(c,full),
                                                      _constant_array(c,useVal),
                                                      _constant_s32_scalar(c,dim0),
                                                      _constant_s32_scalar(c,dim1)
                                                 ),
                                                 xla_client.Shape.array_shape(dtype,(dim0,dim1)))
def Hy5_xla(c,*x,full=False,useVal=useValDefault):
	c = _unpack_builder(c)
	x_shape = c.get_shape(x[0])
	dims = x_shape.dimensions()
	dtype = x_shape.element_type()
	if any(useVal):
		dim0 = dims[0]
	else:
		dim0 = self.basisClass.n
	if full:
		dim1 = self.basisClass.numBasisFuncFull
	else:
		dim1 = self.basisClass.numBasisFunc
	return xla_client.ops.CustomCall(c, xlaName, (_constant_s32_scalar(c,self.basisClass.identifier),
                                                      xla_client.ops.ConcatInDim(c,x,0),
                                                      _constant_array(c,np.array([0,5],dtype=np.int32)),
                                                      _constant_s32_scalar(c,2),
                                                      _constant_bool(c,full),
                                                      _constant_array(c,useVal),
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
xla.backend_specific_translations["cpu"][Hx2y_p] = Hx2y_xla
xla.backend_specific_translations["cpu"][Hx2z_p] = Hx2z_xla
xla.backend_specific_translations["cpu"][Hxy2_p] = Hxy2_xla
xla.backend_specific_translations["cpu"][Hy2z_p] = Hy2z_xla
xla.backend_specific_translations["cpu"][Hx3_p] = Hx3_xla
xla.backend_specific_translations["cpu"][Hy3_p] = Hy3_xla
xla.backend_specific_translations["cpu"][Hz3_p] = Hz3_xla
xla.backend_specific_translations["cpu"][Hxy3_p] = Hxy3_xla
xla.backend_specific_translations["cpu"][Hx3y_p] = Hx3y_xla
xla.backend_specific_translations["cpu"][Hx2y2_p] = Hx2y2_xla
xla.backend_specific_translations["cpu"][Hx4_p] = Hx4_xla
xla.backend_specific_translations["cpu"][Hy4_p] = Hy4_xla
xla.backend_specific_translations["cpu"][Hxy4_p] = Hxy4_xla
xla.backend_specific_translations["cpu"][Hx4y_p] = Hx4y_xla
xla.backend_specific_translations["cpu"][Hx3y2_p] = Hx3y2_xla
xla.backend_specific_translations["cpu"][Hx2y3_p] = Hx2y3_xla
xla.backend_specific_translations["cpu"][Hx5_p] = Hx5_xla
xla.backend_specific_translations["cpu"][Hy5_p] = Hy5_xla

# Batching translations
def H_batch(vec,batch,full=False,useVal=useValDefault):
	return Hjax(*vec,full=full,useVal=useVal), batch[0]
def Hx_batch(vec,batch,full=False,useVal=useValDefault):
	return Hxjax(*vec,full=full,useVal=useVal), batch[0]
def Hy_batch(vec,batch,full=False,useVal=useValDefault):
	return Hyjax(*vec,full=full,useVal=useVal), batch[0]
def Hz_batch(vec,batch,full=False,useVal=useValDefault):
	return Hzjax(*vec,full=full,useVal=useVal), batch[0]
def Hw_batch(vec,batch,full=False,useVal=useValDefault):
	return Hwjax(*vec,full=full,useVal=useVal), batch[0]
def Hxy_batch(vec,batch,full=False,useVal=useValDefault):
	return Hxyjax(*vec,full=full,useVal=useVal), batch[0]
def Hxz_batch(vec,batch,full=False,useVal=useValDefault):
	return Hxzjax(*vec,full=full,useVal=useVal), batch[0]
def Hxw_batch(vec,batch,full=False,useVal=useValDefault):
	return Hxwjax(*vec,full=full,useVal=useVal), batch[0]
def Hyz_batch(vec,batch,full=False,useVal=useValDefault):
	return Hyzjax(*vec,full=full,useVal=useVal), batch[0]
def Hyw_batch(vec,batch,full=False,useVal=useValDefault):
	return Hywjax(*vec,full=full,useVal=useVal), batch[0]
def Hzw_batch(vec,batch,full=False,useVal=useValDefault):
	return Hzwjax(*vec,full=full,useVal=useVal), batch[0]
def Hx2_batch(vec,batch,full=False,useVal=useValDefault):
	return Hx2jax(*vec,full=full,useVal=useVal), batch[0]
def Hy2_batch(vec,batch,full=False,useVal=useValDefault):
	return Hy2jax(*vec,full=full,useVal=useVal), batch[0]
def Hz2_batch(vec,batch,full=False,useVal=useValDefault):
	return Hz2jax(*vec,full=full,useVal=useVal), batch[0]
def Hw2_batch(vec,batch,full=False,useVal=useValDefault):
	return Hw2jax(*vec,full=full,useVal=useVal), batch[0]
def Hx2y_batch(vec,batch,full=False,useVal=useValDefault):
	return Hx2yjax(*vec,full=full,useVal=useVal), batch[0]
def Hx2z_batch(vec,batch,full=False,useVal=useValDefault):
	return Hx2zjax(*vec,full=full,useVal=useVal), batch[0]
def Hxy2_batch(vec,batch,full=False,useVal=useValDefault):
	return Hxy2jax(*vec,full=full,useVal=useVal), batch[0]
def Hy2z_batch(vec,batch,full=False,useVal=useValDefault):
	return Hy2zjax(*vec,full=full,useVal=useVal), batch[0]
def Hx3_batch(vec,batch,full=False,useVal=useValDefault):
	return Hx3jax(*vec,full=full,useVal=useVal), batch[0]
def Hy3_batch(vec,batch,full=False,useVal=useValDefault):
	return Hy3jax(*vec,full=full,useVal=useVal), batch[0]
def Hz3_batch(vec,batch,full=False,useVal=useValDefault):
	return Hz3jax(*vec,full=full,useVal=useVal), batch[0]
def Hxy3_batch(vec,batch,full=False,useVal=useValDefault):
	return Hxy3jax(*vec,full=full,useVal=useVal), batch[0]
def Hx3y_batch(vec,batch,full=False,useVal=useValDefault):
	return Hx3yjax(*vec,full=full,useVal=useVal), batch[0]
def Hx2y2_batch(vec,batch,full=False,useVal=useValDefault):
	return Hx2y2jax(*vec,full=full,useVal=useVal), batch[0]
def Hx4_batch(vec,batch,full=False,useVal=useValDefault):
	return Hx4jax(*vec,full=full,useVal=useVal), batch[0]
def Hy4_batch(vec,batch,full=False,useVal=useValDefault):
	return Hy4jax(*vec,full=full,useVal=useVal), batch[0]
def Hxy4_batch(vec,batch,full=False,useVal=useValDefault):
	return Hxy4jax(*vec,full=full,useVal=useVal), batch[0]
def Hx4y_batch(vec,batch,full=False,useVal=useValDefault):
	return Hx4yjax(*vec,full=full,useVal=useVal), batch[0]
def Hx3y2_batch(vec,batch,full=False,useVal=useValDefault):
	return Hx3y2jax(*vec,full=full,useVal=useVal), batch[0]
def Hx2y3_batch(vec,batch,full=False,useVal=useValDefault):
	return Hx2y3jax(*vec,full=full,useVal=useVal), batch[0]
def Hx5_batch(vec,batch,full=False,useVal=useValDefault):
	return Hx5jax(*vec,full=full,useVal=useVal), batch[0]
def Hy5_batch(vec,batch,full=False,useVal=useValDefault):
	return Hy5jax(*vec,full=full,useVal=useVal), batch[0]

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
batching.primitive_batchers[Hx2y_p] = Hx2y_batch
batching.primitive_batchers[Hx2z_p] = Hx2z_batch
batching.primitive_batchers[Hxy2_p] = Hxy2_batch
batching.primitive_batchers[Hy2z_p] = Hy2z_batch
batching.primitive_batchers[Hx3_p] = Hx3_batch
batching.primitive_batchers[Hy3_p] = Hy3_batch
batching.primitive_batchers[Hz3_p] = Hz3_batch
batching.primitive_batchers[Hxy3_p] = Hxy3_batch
batching.primitive_batchers[Hx3y_p] = Hx3y_batch
batching.primitive_batchers[Hx2y2_p] = Hx2y2_batch
batching.primitive_batchers[Hx4_p] = Hx4_batch
batching.primitive_batchers[Hy4_p] = Hy4_batch
batching.primitive_batchers[Hxy4_p] = Hxy4_batch
batching.primitive_batchers[Hx4y_p] = Hx4y_batch
batching.primitive_batchers[Hx3y2_p] = Hx3y2_batch
batching.primitive_batchers[Hx2y3_p] = Hx2y3_batch
batching.primitive_batchers[Hx5_p] = Hx5_batch
batching.primitive_batchers[Hy5_p] = Hy5_batch

# Jacobian vector translations
def H_jvp(arg_vals,arg_tans,full=False,useVal=useValDefault):
	funcs = [Hxjax,Hyjax,Hzjax,Hwjax]
	n = min(len(arg_vals),len(funcs))
	flat = len(arg_vals[0].shape) == 1
	if any(useVal):
		dim0 = arg_vals[0].shape[0]
	else:
		dim0 = self.basisClass.n
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
					out_tans += funcs[k](*arg_vals,full=full,useVal=useVal)*np.expand_dims(arg_tans[k],1)
				else:
					out_tans += funcs[k](*arg_vals,full=full,useVal=useVal)*arg_tans[k]
	return (Hjax(*arg_vals,full=full,useVal=useVal),out_tans)
def Hx_jvp(arg_vals,arg_tans,full=False,useVal=useValDefault):
	funcs = [Hx2jax,Hxyjax,Hxzjax,Hxwjax]
	n = min(len(arg_vals),len(funcs))
	flat = len(arg_vals[0].shape) == 1
	if any(useVal):
		dim0 = arg_vals[0].shape[0]
	else:
		dim0 = self.basisClass.n
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
					out_tans += funcs[k](*arg_vals,full=full,useVal=useVal)*np.expand_dims(arg_tans[k],1)
				else:
					out_tans += funcs[k](*arg_vals,full=full,useVal=useVal)*arg_tans[k]
	return (Hxjax(*arg_vals,full=full,useVal=useVal),out_tans)
def Hy_jvp(arg_vals,arg_tans,full=False,useVal=useValDefault):
	funcs = [Hxyjax,Hy2jax,Hyzjax,Hywjax]
	n = min(len(arg_vals),len(funcs))
	flat = len(arg_vals[0].shape) == 1
	if any(useVal):
		dim0 = arg_vals[0].shape[0]
	else:
		dim0 = self.basisClass.n
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
					out_tans += funcs[k](*arg_vals,full=full,useVal=useVal)*np.expand_dims(arg_tans[k],1)
				else:
					out_tans += funcs[k](*arg_vals,full=full,useVal=useVal)*arg_tans[k]
	return (Hyjax(*arg_vals,full=full,useVal=useVal),out_tans)
def Hz_jvp(arg_vals,arg_tans,full=False,useVal=useValDefault):
	funcs = [Hxzjax,Hyzjax,Hz2jax,Hzwjax]
	n = min(len(arg_vals),len(funcs))
	flat = len(arg_vals[0].shape) == 1
	if any(useVal):
		dim0 = arg_vals[0].shape[0]
	else:
		dim0 = self.basisClass.n
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
					out_tans += funcs[k](*arg_vals,full=full,useVal=useVal)*np.expand_dims(arg_tans[k],1)
				else:
					out_tans += funcs[k](*arg_vals,full=full,useVal=useVal)*arg_tans[k]
	return (Hzjax(*arg_vals,full=full,useVal=useVal),out_tans)
def Hw_jvp(arg_vals,arg_tans,full=False,useVal=useValDefault):
	funcs = [Hxwjax,Hywjax,Hzwjax,Hw2jax]
	n = min(len(arg_vals),len(funcs))
	flat = len(arg_vals[0].shape) == 1
	if any(useVal):
		dim0 = arg_vals[0].shape[0]
	else:
		dim0 = self.basisClass.n
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
					out_tans += funcs[k](*arg_vals,full=full,useVal=useVal)*np.expand_dims(arg_tans[k],1)
				else:
					out_tans += funcs[k](*arg_vals,full=full,useVal=useVal)*arg_tans[k]
	return (Hwjax(*arg_vals,full=full,useVal=useVal),out_tans)
def Hxy_jvp(arg_vals,arg_tans,full=False,useVal=useValDefault):
	funcs = [Hx2yjax,Hxy2jax]
	n = min(len(arg_vals),len(funcs))
	flat = len(arg_vals[0].shape) == 1
	if any(useVal):
		dim0 = arg_vals[0].shape[0]
	else:
		dim0 = self.basisClass.n
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
					out_tans += funcs[k](*arg_vals,full=full,useVal=useVal)*np.expand_dims(arg_tans[k],1)
				else:
					out_tans += funcs[k](*arg_vals,full=full,useVal=useVal)*arg_tans[k]
	return (Hxyjax(*arg_vals,full=full,useVal=useVal),out_tans)
def Hxz_jvp(arg_vals,arg_tans,full=False,useVal=useValDefault):
	funcs = [Hx2zjax]
	n = min(len(arg_vals),len(funcs))
	flat = len(arg_vals[0].shape) == 1
	if any(useVal):
		dim0 = arg_vals[0].shape[0]
	else:
		dim0 = self.basisClass.n
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
					out_tans += funcs[k](*arg_vals,full=full,useVal=useVal)*np.expand_dims(arg_tans[k],1)
				else:
					out_tans += funcs[k](*arg_vals,full=full,useVal=useVal)*arg_tans[k]
	return (Hxzjax(*arg_vals,full=full,useVal=useVal),out_tans)
def Hyz_jvp(arg_vals,arg_tans,full=False,useVal=useValDefault):
	funcs = [Hy2zjax]
	n = min(len(arg_vals),len(funcs))
	flat = len(arg_vals[0].shape) == 1
	if any(useVal):
		dim0 = arg_vals[0].shape[0]
	else:
		dim0 = self.basisClass.n
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
					out_tans += funcs[k](*arg_vals,full=full,useVal=useVal)*np.expand_dims(arg_tans[k],1)
				else:
					out_tans += funcs[k](*arg_vals,full=full,useVal=useVal)*arg_tans[k]
	return (Hyzjax(*arg_vals,full=full,useVal=useVal),out_tans)
def Hx2_jvp(arg_vals,arg_tans,full=False,useVal=useValDefault):
	funcs = [Hx3jax,Hx2yjax]
	n = min(len(arg_vals),len(funcs))
	flat = len(arg_vals[0].shape) == 1
	if any(useVal):
		dim0 = arg_vals[0].shape[0]
	else:
		dim0 = self.basisClass.n
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
					out_tans += funcs[k](*arg_vals,full=full,useVal=useVal)*np.expand_dims(arg_tans[k],1)
				else:
					out_tans += funcs[k](*arg_vals,full=full,useVal=useVal)*arg_tans[k]
	return (Hx2jax(*arg_vals,full=full,useVal=useVal),out_tans)
def Hy2_jvp(arg_vals,arg_tans,full=False,useVal=useValDefault):
	funcs = [Hxy2jax,Hy3jax]
	n = min(len(arg_vals),len(funcs))
	flat = len(arg_vals[0].shape) == 1
	if any(useVal):
		dim0 = arg_vals[0].shape[0]
	else:
		dim0 = self.basisClass.n
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
					out_tans += funcs[k](*arg_vals,full=full,useVal=useVal)*np.expand_dims(arg_tans[k],1)
				else:
					out_tans += funcs[k](*arg_vals,full=full,useVal=useVal)*arg_tans[k]
	return (Hy2jax(*arg_vals,full=full,useVal=useVal),out_tans)
def Hz2_jvp(arg_vals,arg_tans,full=False,useVal=useValDefault):
	funcs = [Hz3jax]
	n = min(len(arg_vals),len(funcs))
	flat = len(arg_vals[0].shape) == 1
	if any(useVal):
		dim0 = arg_vals[0].shape[0]
	else:
		dim0 = self.basisClass.n
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
					out_tans += funcs[k](*arg_vals,full=full,useVal=useVal)*np.expand_dims(arg_tans[k],1)
				else:
					out_tans += funcs[k](*arg_vals,full=full,useVal=useVal)*arg_tans[k]
	return (Hz2jax(*arg_vals,full=full,useVal=useVal),out_tans)
def Hx2y_jvp(arg_vals,arg_tans,full=False,useVal=useValDefault):
	funcs = [Hx3yjax,Hx2y2jax]
	n = min(len(arg_vals),len(funcs))
	flat = len(arg_vals[0].shape) == 1
	if any(useVal):
		dim0 = arg_vals[0].shape[0]
	else:
		dim0 = self.basisClass.n
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
					out_tans += funcs[k](*arg_vals,full=full,useVal=useVal)*np.expand_dims(arg_tans[k],1)
				else:
					out_tans += funcs[k](*arg_vals,full=full,useVal=useVal)*arg_tans[k]
	return (Hx2yjax(*arg_vals,full=full,useVal=useVal),out_tans)
def Hxy2_jvp(arg_vals,arg_tans,full=False,useVal=useValDefault):
	funcs = [Hx2y2jax,Hxy3jax]
	n = min(len(arg_vals),len(funcs))
	flat = len(arg_vals[0].shape) == 1
	if any(useVal):
		dim0 = arg_vals[0].shape[0]
	else:
		dim0 = self.basisClass.n
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
					out_tans += funcs[k](*arg_vals,full=full,useVal=useVal)*np.expand_dims(arg_tans[k],1)
				else:
					out_tans += funcs[k](*arg_vals,full=full,useVal=useVal)*arg_tans[k]
	return (Hxy2jax(*arg_vals,full=full,useVal=useVal),out_tans)
def Hx3_jvp(arg_vals,arg_tans,full=False,useVal=useValDefault):
	funcs = [Hx4jax,Hx3yjax]
	n = min(len(arg_vals),len(funcs))
	flat = len(arg_vals[0].shape) == 1
	if any(useVal):
		dim0 = arg_vals[0].shape[0]
	else:
		dim0 = self.basisClass.n
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
					out_tans += funcs[k](*arg_vals,full=full,useVal=useVal)*np.expand_dims(arg_tans[k],1)
				else:
					out_tans += funcs[k](*arg_vals,full=full,useVal=useVal)*arg_tans[k]
	return (Hx3jax(*arg_vals,full=full,useVal=useVal),out_tans)
def Hy3_jvp(arg_vals,arg_tans,full=False,useVal=useValDefault):
	funcs = [Hxy3jax,Hy4jax]
	n = min(len(arg_vals),len(funcs))
	flat = len(arg_vals[0].shape) == 1
	if any(useVal):
		dim0 = arg_vals[0].shape[0]
	else:
		dim0 = self.basisClass.n
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
					out_tans += funcs[k](*arg_vals,full=full,useVal=useVal)*np.expand_dims(arg_tans[k],1)
				else:
					out_tans += funcs[k](*arg_vals,full=full,useVal=useVal)*arg_tans[k]
	return (Hy3jax(*arg_vals,full=full,useVal=useVal),out_tans)
def Hxy3_jvp(arg_vals,arg_tans,full=False,useVal=useValDefault):
	funcs = [Hx2y3jax,Hxy4jax]
	n = min(len(arg_vals),len(funcs))
	flat = len(arg_vals[0].shape) == 1
	if any(useVal):
		dim0 = arg_vals[0].shape[0]
	else:
		dim0 = self.basisClass.n
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
					out_tans += funcs[k](*arg_vals,full=full,useVal=useVal)*np.expand_dims(arg_tans[k],1)
				else:
					out_tans += funcs[k](*arg_vals,full=full,useVal=useVal)*arg_tans[k]
	return (Hxy3jax(*arg_vals,full=full,useVal=useVal),out_tans)
def Hx3y_jvp(arg_vals,arg_tans,full=False,useVal=useValDefault):
	funcs = [Hx4yjax,Hx3y2jax]
	n = min(len(arg_vals),len(funcs))
	flat = len(arg_vals[0].shape) == 1
	if any(useVal):
		dim0 = arg_vals[0].shape[0]
	else:
		dim0 = self.basisClass.n
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
					out_tans += funcs[k](*arg_vals,full=full,useVal=useVal)*np.expand_dims(arg_tans[k],1)
				else:
					out_tans += funcs[k](*arg_vals,full=full,useVal=useVal)*arg_tans[k]
	return (Hx3yjax(*arg_vals,full=full,useVal=useVal),out_tans)
def Hx2y2_jvp(arg_vals,arg_tans,full=False,useVal=useValDefault):
	funcs = [Hx3y2jax,Hx2y3jax]
	n = min(len(arg_vals),len(funcs))
	flat = len(arg_vals[0].shape) == 1
	if any(useVal):
		dim0 = arg_vals[0].shape[0]
	else:
		dim0 = self.basisClass.n
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
					out_tans += funcs[k](*arg_vals,full=full,useVal=useVal)*np.expand_dims(arg_tans[k],1)
				else:
					out_tans += funcs[k](*arg_vals,full=full,useVal=useVal)*arg_tans[k]
	return (Hx2y2jax(*arg_vals,full=full,useVal=useVal),out_tans)
def Hx4_jvp(arg_vals,arg_tans,full=False,useVal=useValDefault):
	funcs = [Hx5jax,Hx4yjax]
	n = min(len(arg_vals),len(funcs))
	flat = len(arg_vals[0].shape) == 1
	if any(useVal):
		dim0 = arg_vals[0].shape[0]
	else:
		dim0 = self.basisClass.n
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
					out_tans += funcs[k](*arg_vals,full=full,useVal=useVal)*np.expand_dims(arg_tans[k],1)
				else:
					out_tans += funcs[k](*arg_vals,full=full,useVal=useVal)*arg_tans[k]
	return (Hx4jax(*arg_vals,full=full,useVal=useVal),out_tans)
def Hy4_jvp(arg_vals,arg_tans,full=False,useVal=useValDefault):
	funcs = [Hxy4jax,Hy5jax]
	n = min(len(arg_vals),len(funcs))
	flat = len(arg_vals[0].shape) == 1
	if any(useVal):
		dim0 = arg_vals[0].shape[0]
	else:
		dim0 = self.basisClass.n
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
					out_tans += funcs[k](*arg_vals,full=full,useVal=useVal)*np.expand_dims(arg_tans[k],1)
				else:
					out_tans += funcs[k](*arg_vals,full=full,useVal=useVal)*arg_tans[k]
	return (Hy4jax(*arg_vals,full=full,useVal=useVal),out_tans)
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
ad.primitive_jvps[Hx2y_p] = Hx2y_jvp
ad.primitive_jvps[Hxy2_p] = Hxy2_jvp
ad.primitive_jvps[Hx3_p] = Hx3_jvp
ad.primitive_jvps[Hy3_p] = Hy3_jvp
ad.primitive_jvps[Hxy3_p] = Hxy3_jvp
ad.primitive_jvps[Hx3y_p] = Hx3y_jvp
ad.primitive_jvps[Hx2y2_p] = Hx2y2_jvp
ad.primitive_jvps[Hx4_p] = Hx4_jvp
ad.primitive_jvps[Hy4_p] = Hy4_jvp

