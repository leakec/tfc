#include <iostream>
#include <math.h>
#include <Python.h>
#include "BF.h"

// COP: **********************************************************************
CP::CP(double* zin, int zDim0, int* nCin, int ncDim0, int min, double cin){
	z = new double[zDim0];
	memcpy(z,zin,zDim0*sizeof(double));

	//nC = new int[ncDim0];
	//memcpy(nC,nCin,ncDim0*sizeof(int));

	numC = ncDim0;

	n = zDim0;
	m = min;
	c = cin;
};

CP::~CP(){
	delete CP::z;
	//delete CP::nC;
};

void CP::H(double* x, int in, const int d, int* nOut, int* mOut, double** F,  bool full, bool useVal){
	*mOut = full ? m : m-numC;

	if (useVal){
		*nOut = n;
		*F = new double[mOut[0]*nOut[0]];
		Hint(d,x,*nOut,*mOut,*F,full);
	} else { 
		*nOut = in;
		*F = new double[mOut[0]*nOut[0]];
		Hint(d,z,*nOut,*mOut,*F,full);
		Hint(d,x,*nOut,*mOut,*F,full);
	}
};

void CP::xla(void* out, void** in){
	double* out_buf = reinterpret_cast<double*>(out);
	bool useVal = (reinterpret_cast<bool*>(in[5]))[0];
	double* x = useVal ? reinterpret_cast<double*>(in[0]) : z;
	int nOut = (reinterpret_cast<int*>(in[1]))[0];
	int mOut = (reinterpret_cast<int*>(in[2]))[0];
	int d = (reinterpret_cast<int*>(in[3]))[0];
	bool full = (reinterpret_cast<bool*>(in[4]))[0];
	CP::Hint(d,x,nOut,mOut,out_buf,full);
};

PyObject* CP::GetXlaCapsule(){
	void (*fncPtr)(void*, void**) = &CP::xla;
	const char* name = "xla._CUSTOM_CALL_TARGET";
	PyObject* capsule;
	capsule = PyCapsule_New(reinterpret_cast<void*>(fncPtr), name, NULL);
	return capsule;
};

void CP::Hint(const int d, const double* x, const int nOut, const int mOut, double*& F, bool full){

	int j,k;
	int deg = m-1;
	double dMult = pow(CP::c,d);
	double * dark = new double[CP::m*nOut];
	if (deg == 0){
		if (d >0){
			for (k=0;k<nOut;k++)
				dark[k] = 0.;
		} else {
			for (k=0;k<nOut;k++)
				dark[k] = 1.;
		}
	} else if (deg == 1){
		if (d > 1){
			for (k=0;k<m*nOut;k++)
				dark[k] = 0.;
		} else if (d > 0){
			for (k=0;k<nOut;k++){
				dark[m*k] = 0.;
				dark[m*k+1] = 1.;
			}
		} else {
			for (k=0;k<nOut;k++){
				dark[m*k] = 1;
				dark[m*k+1] = x[k];
			}
		}
	} else {
		for (k=0;k<nOut;k++){
			dark[m*k] = 1;
			dark[m*k+1] = x[k];
		}
		for (k=2;k<m;k++){
			for (j=0;j<nOut;j++)
				dark[m*j+k] = 2.*x[j]*dark[m*j+k-1]-dark[m*j+k-2];
		}
		CP::RecurseDeriv(d,0,x,nOut,dark,m);
	}
	//if (!full){
	//	int i=0;
	//	bool flag;
	//	for (j=0;j<m;j++){
	//		flag = false;
	//		for (k=0;k<numC;k++){
	//			if (j == nC[k]){
	//				flag = true;
	//				break;
	//			}
	//		}
	//		if (flag) continue; else i++;
	//		for (k=0;k<nOut;k++)
	//			F[mOut*k+i] = dark[m*k+j]*dMult;
	//	}
	//} else {
	//	for (j=0;j<m;j++){
	//		for (k=0;k<nOut;k++)
	//			F[m*k+j] = dark[m*k+j]*dMult;
	//	}
	//}
	for (j=0;j<m;j++){
		for (k=0;k<nOut;k++)
			F[m*k+j] = dark[m*k+j]*dMult;
	}
	return;
};

void CP::RecurseDeriv(const int d, int dCurr, const double* x, const int nOut, double*& F, const int mOut){

	int j, k;
	double* dark = F;
	if (dCurr != d){
		F = new double[mOut*nOut]{0};
		if (dCurr == 0){
			for (k=0;k<nOut;k++)
				F[mOut*k+1] = 1;
		}
		for (k=2;k<mOut;k++){
			for (j=0;j<nOut;j++)
				F[mOut*j+k] = (2.+2.*dCurr)*dark[mOut*j+k-1]+2.*x[j]*F[mOut*j+k-1]-F[mOut*j+k-2];
		}
		dCurr++;
		delete dark;
		CP::RecurseDeriv(d,dCurr,x,nOut,F,mOut);
	}
	return;
};


// Save for nCP: ********************************************************************************

//void CP::nH(double* x, int in, int dimx, int min, int *d, int dDim, int* nC, int nDim, double* c, int inc, int* n, int* m, double**F){
//	*n = in;
//	*m = nHint(d,c,x,dimx,in,min,nC,*F);
//};

//int CP::nHint(const int* d, const double* c, const double *x, const int dim, const int n, const int m, const int* nC, double*& F){
//	int j,k,numBasis=0,count=0;
//	double* T = new double[n*m*dim];
//	double* dark = NULL;
//	double* dark1 = new double[n];
//	double dark2 = 0.;
//	int* vec = new int[dim]{0};
//
//	for (k=0;k<dim;k++){
//		for (j=0;j<n;j++)
//			dark1[j] = x[j*dim+k];
//		CP::Hint(d[k],dark1,n,m,dark);
//		dark2 = pow(c[k],d[k]);
//		for (j=0;j<n*m;j++)
//			T[j+k*m*n] = dark[j]*dark2;
//		free(dark);
//	}
//
//	nRecurseBasis(nC,m,dim,dim-1,vec,numBasis);
//	F = new double[numBasis*n];
//	for (k=0;k<n*numBasis;k++)
//		F[k] = 1.;
//	nRecurse(nC,m,n,dim,numBasis,T,dim-1,vec,count,F);
//	delete [] T;
//	delete [] dark1;
//	delete [] vec;
//	return numBasis;
//};

