# At the end do expand -4 JaxPrintout.py > test.py
# This is utiliy used for creating the jax primitives for nTFC.
import numpy as np

# Printout file
fileName = "JaxPrimitivePrintout.py"

prims = {"H":{'deriv':"np.array([0],dtype=np.int32)",'derivFuncs':["Hx","Hy","Hz","Hw"]},
         "Hx":{'deriv':"np.array([1],dtype=np.int32)",'derivFuncs':["Hx2","Hxy","Hxz","Hxw"]},
         "Hy":{'deriv':"np.array([0,1],dtype=np.int32)",'derivFuncs':["Hxy","Hy2","Hyz","Hyw"]},
         "Hz":{'deriv':"np.array([0,0,1],dtype=np.int32)",'derivFuncs':["Hxz","Hyz","Hz2","Hzw"]},
         "Hw":{'deriv':"np.array([0,0,0,1],dtype=np.int32)",'derivFuncs':["Hxw","Hyw","Hzw","Hw2"]},
         "Hxy":{'deriv':"np.array([1,1],dtype=np.int32)",'derivFuncs':["Hx2y","Hxy2"]},
         "Hxz":{'deriv':"np.array([1,0,1],dtype=np.int32)",'derivFuncs':["Hx2z"]},
         "Hxw":{'deriv':"np.array([1,0,0,1],dtype=np.int32)",'derivFuncs':[None]},
         "Hyz":{'deriv':"np.array([0,1,1],dtype=np.int32)",'derivFuncs':[None,"Hy2z"]},
         "Hyw":{'deriv':"np.array([0,1,0,1],dtype=np.int32)",'derivFuncs':[None]},
         "Hzw":{'deriv':"np.array([0,0,1,1],dtype=np.int32)",'derivFuncs':[None]},
         "Hx2":{'deriv':"np.array([2],dtype=np.int32)",'derivFuncs':["Hx3","Hx2y"]},
         "Hy2":{'deriv':"np.array([0,2],dtype=np.int32)",'derivFuncs':["Hxy2","Hy3"]},
         "Hz2":{'deriv':"np.array([0,0,2],dtype=np.int32)",'derivFuncs':[None,None,"Hz3"]},
         "Hw2":{'deriv':"np.array([0,0,0,2],dtype=np.int32)",'derivFuncs':[None]},
         "Hx2y":{'deriv':"np.array([2,1],dtype=np.int32)",'derivFuncs':["Hx3y","Hx2y2"]},
         "Hx2z":{'deriv':"np.array([2,0,1],dtype=np.int32)",'derivFuncs':[None]},
         "Hxy2":{'deriv':"np.array([1,2],dtype=np.int32)",'derivFuncs':["Hx2y2","Hxy3"]},
         "Hy2z":{'deriv':"np.array([0,2,1],dtype=np.int32)",'derivFuncs':[None]},
         "Hx3":{'deriv':"np.array([3],dtype=np.int32)",'derivFuncs':["Hx4","Hx3y"]},
         "Hy3":{'deriv':"np.array([0,3],dtype=np.int32)",'derivFuncs':["Hxy3","Hy4"]},
         "Hz3":{'deriv':"np.array([0,0,3],dtype=np.int32)",'derivFuncs':[None]},
         "Hxy3":{'deriv':"np.array([1,3],dtype=np.int32)",'derivFuncs':["Hx2y3","Hxy4"]},
         "Hx3y":{'deriv':"np.array([3,1],dtype=np.int32)",'derivFuncs':["Hx4y","Hx3y2"]},
         "Hx2y2":{'deriv':"np.array([2,2],dtype=np.int32)",'derivFuncs':["Hx3y2","Hx2y3"]},
         "Hx4":{'deriv':"np.array([4],dtype=np.int32)",'derivFuncs':["Hx5","Hx4y"]},
         "Hy4":{'deriv':"np.array([0,4],dtype=np.int32)",'derivFuncs':["Hxy4","Hy5"]},
         "Hxy4":{'deriv':"np.array([1,4],dtype=np.int32)",'derivFuncs':[None]},
         "Hx4y":{'deriv':"np.array([4,1],dtype=np.int32)",'derivFuncs':[None]},
         "Hx3y2":{'deriv':"np.array([3,2],dtype=np.int32)",'derivFuncs':[None]},
         "Hx2y3":{'deriv':"np.array([2,3],dtype=np.int32)",'derivFuncs':[None]},
         "Hx5":{'deriv':"np.array([5],dtype=np.int32)",'derivFuncs':[None]},
         "Hy5":{'deriv':"np.array([0,5],dtype=np.int32)",'derivFuncs':[None]}}

# Names of all the primitives
names = list(prims.keys())

# Variable names used
varNames = ['x','y','z','w']

# Open the file for writing
fid = open(fileName,"w")

# Constants
n = len(names)

# Create primitives
fid.write("useValDefault = self.useValDefault\n\n")
fid.write("# Create Primitives\n")
for k in range(n):
    fid.write(names[k]+'_p = core.Primitive("'+names[k]+'")\n')
fid.write("\n")
    
for k in range(n):
    fid.write('def '+names[k]+'jax(*x,full=False,useVal=useValDefault):\n\treturn '+names[k]+'_p.bind(*x,full=full,useVal=useVal)\n')
fid.write("\n")

# Create implicit translations
fid.write("# Implicit translations\n")
for k in range(n):
    fid.write('def '+names[k]+'_impl(*x,full=False,useVal=useValDefault):\n\treturn self.basisClass.H(np.array(x),'+prims[names[k]]['deriv']+',full,useVal)\n')
fid.write("\n")

for k in range(n):
    fid.write(names[k]+'_p.def_impl('+names[k]+'_impl)\n')
fid.write("\n")

# Abstract evaluation
fid.write("def H_abstract_eval(*x,full=False,useVal=useValDefault):\n\tif any(useVal):\n\t\tdim0 = x[0].shape[0]\n\telse:\n\t\tdim0 = self.basisClass.n\n\tif full:\n\t\tdim1 = self.basisClass.numBasisFuncFull\n\telse:\n\t\tdim1 = self.basisClass.numBasisFunc\n\treturn abstract_arrays.ShapedArray((dim0,dim1),x[0].dtype)\n")
fid.write("\n")

for k in range(n):
    fid.write(names[k]+'_p.def_abstract_eval(H_abstract_eval)\n')
fid.write("\n")

# XLA compilation
def CreateXlaString(k):
    lenDeriv = eval(prims[names[k]]['deriv']).shape[0]
    mystr = "def "+names[k]+"_xla(c,*x,full=False,useVal=useValDefault):\n\tc = _unpack_builder(c)\n\tx_shape = c.get_shape(x[0])\n\tdims = x_shape.dimensions()\n\tdtype = x_shape.element_type()\n\tif any(useVal):\n\t\tdim0 = dims[0]\n\telse:\n\t\tdim0 = self.basisClass.n\n\tif full:\n\t\tdim1 = self.basisClass.numBasisFuncFull\n\telse:\n\t\tdim1 = self.basisClass.numBasisFunc\n\t"+"return xla_client.ops.CustomCall(c, xlaName, (_constant_s32_scalar(c,self.basisClass.identifier),\n\
                                                      xla_client.ops.ConcatInDim(c,x,0),\n\
                                                      _constant_array(c,"+prims[names[k]]['deriv']+"),\n\
                                                      _constant_s32_scalar(c,"+str(lenDeriv)+"),\n\
                                                      _constant_bool(c,full),\n\
                                                      _constant_array(c,useVal),\n\
                                                      _constant_s32_scalar(c,dim0),\n\
                                                      _constant_s32_scalar(c,dim1)\n\
                                                 ),\n\
                                                 xla_client.Shape.array_shape(dtype,(dim0,dim1)))\n"
    return mystr

fid.write("# XLA compilation\n")
for k in range(n):
    fid.write(CreateXlaString(k))
fid.write("\n")

for k in range(n):
    fid.write('xla.backend_specific_translations["cpu"]['+names[k]+'_p] = '+names[k]+'_xla\n')
fid.write("\n")

# Batching translations
fid.write("# Batching translations\n")

for k in range(n):
    fid.write("def "+names[k]+"_batch(vec,batch,full=False,useVal=useValDefault):\n\treturn "+names[k]+"jax(*vec,full=full,useVal=useVal), batch[0]\n")
fid.write("\n")

for k in range(n):
    fid.write("batching.primitive_batchers["+names[k]+"_p] = "+names[k]+"_batch\n")
fid.write("\n")

# Jacobian vector translations
fid.write("# Jacobian vector translations\n")

for k in range(n):
    if all([g is None for g in prims[names[k]]['derivFuncs']]):
        continue
    funcsCurr = prims[names[k]]['derivFuncs']
    funcs = ""
    for g in range(len(funcsCurr)):
        if not funcsCurr[g] is None:
            funcs += funcsCurr[g]+'jax,'
    funcs = funcs[:-1]
    fid.write("def "+names[k]+"_jvp(arg_vals,arg_tans,full=False,useVal=useValDefault):\n\tfuncs = ["+funcs+"]\n\tn = min(len(arg_vals),len(funcs))\n\tflat = len(arg_vals[0].shape) == 1\n\tif any(useVal):\n\t\tdim0 = arg_vals[0].shape[0]\n\telse:\n\t\tdim0 = self.basisClass.n\n\tif full:\n\t\tdim1 = self.basisClass.numBasisFuncFull\n\telse:\n\t\tdim1 = self.basisClass.numBasisFunc\n\tout_tans = np.zeros((dim0,dim1))\n\tfor k in range(n):\n\t\tif not (type(arg_tans[k]) is ad.Zero):\n\t\t\tif type(arg_tans[k]) is batching.BatchTracer:\n\t\t\t\tflag = onp.any(arg_tans[k].val != 0)\n\t\t\telse:\n\t\t\t\tflag = onp.any(arg_tans[k] != 0)\n\t\t\tif flag:\n\t\t\t\tif flat:\n\t\t\t\t\tout_tans += funcs[k](*arg_vals,full=full,useVal=useVal)*np.expand_dims(arg_tans[k],1)\n\t\t\t\telse:\n\t\t\t\t\tout_tans += funcs[k](*arg_vals,full=full,useVal=useVal)*arg_tans[k]\n\treturn ("+names[k]+"jax(*arg_vals,full=full,useVal=useVal),out_tans)\n")

for k in range(n):
    if all([g is None for g in prims[names[k]]['derivFuncs']]):
        continue
    fid.write("ad.primitive_jvps["+names[k]+"_p] = "+names[k]+"_jvp\n")
fid.write("\n")

# Close the file
fid.close()
