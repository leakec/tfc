// BF.i

%module BF
%{
#define SWIG_FILE_WITH_INIT
#include <iostream>
#include <vector>
#include <math.h>
#include <Python.h>
#ifdef HAS_CUDA
    #include <cuda.h>
    #include <cuda_runtime.h>
    #include <cuda_runtime_api.h>
#endif
#include "BF.h"
%}

%include "numpy.i"
%include <typemaps.i>
%include <attribute.i>
%apply bool* INPUT {bool* useVal};

%init %{
        import_array();
%}
%ignore xlaGpuWrapper(CUstream stream, void** buffers, const char* opaque, size_t opaque_len);

// Apply typemaps to allow hooks into Python
%apply (int* IN_ARRAY1, int DIM1){(int* d, int dDim0),(int* nCin, int ncDim0),(int* useVal, int useValDim0)};
%apply (double* IN_ARRAY1, int DIM1){(double* x, int n),(double* cin, int cDim0),(double* arrIn, int nIn),(double* zin, int zDim0),(double* x0in, int x0Dim0),(double* xf, int xfDim0)};
%apply (int* IN_ARRAY2, int DIM1, int DIM2){(int* nCin, int ncDim0, int ncDim1)};
%apply (double* IN_ARRAY2, int DIM1, int DIM2){(double* zin, int zDim0, int zDim1),(double* x, int in, int xDim1),(double* arrIn, int dimIn, int nIn)};

%apply (int* DIM1, int* DIM2, double** ARGOUTVIEWM_ARRAY2){(int* nOut, int* mOut, double** F),(int* dimOut, int* nOut, double** arrOut)}; // Switch to ARGOUTVIEWM when you can to avoid memory leaks
%apply (double** ARGOUTVIEWM_ARRAY1, int* DIM1){(double** arrOut, int* nOut)};

// Add getter and setter methods 

%extend nBasisFunc{
    %rename(_getC) getC(double** arrOut, int* nOut);

    %pythoncode %{
        c = property(_getC)
    %}
};

%extend ELM{
    %rename(_getW) getW(double** arrOut, int* nOut);
    %rename(_setW) setW(double* arrIn, int nIn);
    %rename(_getB) getB(double** arrOut, int* nOut);
    %rename(_setB) setB(double* arrIn, int nIn);
    %ignore w;
    %ignore b;

    %pythoncode %{
        w = property(_getW,_setW)
        b = property(_getB,_setB)
    %}

};

%extend nELM{
    %rename(_getW) getW(int* dimOut, int* nOut, double** arrOut);
    %rename(_setW) setW(double* arrIn, int dimIn, int nIn);
    %rename(_getB) getB(double** arrOut, int* nOut);
    %rename(_setB) setB(double* arrIn, int nIn);
    %ignore w;
    %ignore b;

    %pythoncode %{
        w = property(_getW,_setW)
        b = property(_getB,_setB)
    %}

};

%include "BF.h"
