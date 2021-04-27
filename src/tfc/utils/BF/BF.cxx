#include <iostream>
#include <vector>
#include <math.h>
#include <float.h>
#include <Python.h>
#ifdef HAS_CUDA
	#include <cuda.h>
	#include <cuda_runtime.h>
	#include <cuda_runtime_api.h>
#endif
#include "BF.h"

// Initialize static BasisFunc variables
int BasisFunc::nIdentifier = 0;
std::vector<BasisFunc*> BasisFunc::BasisFuncContainer;

// xlaWrapper function
void xlaWrapper(void* out, void** in){
	int N = (reinterpret_cast<int*>(in[0]))[0];
	BasisFunc::BasisFuncContainer[N]->xla(out,in);
};

#ifdef HAS_CUDA
	// xlaGpuWrapper function
	void xlaGpuWrapper(CUstream stream, void** buffers, const char* opaque, size_t opaque_len){
		int* N = new int[1];
		N[0] = 0;
		//cudaMemcpy(N,reinterpret_cast<int*>(buffers[6]),1*sizeof(int),cudaMemcpyDeviceToHost);
		BasisFunc::BasisFuncContainer[*N]->xlaGpu(stream,buffers,opaque,opaque_len);
		delete[] N;
	};
#endif

// Parent basis function class: **********************************************************************
BasisFunc::BasisFunc(double x0in, double xf, int* nCin, int ncDim0, int min, double z0in, double zf){

	// Initialize internal variables based on user givens
	nC = new int[ncDim0];
	memcpy(nC,nCin,ncDim0*sizeof(int));
	numC = ncDim0;

	z0 = z0in;

	m = min;
	if (zf == DBL_MAX){
		c = 1.; x0 = 0.;
	} else {
		x0 = x0in;
		c = (zf-z0)/(xf-x0);
	}

	// Track this instance of BasisFunc 
	BasisFuncContainer.push_back(this);
	identifier = nIdentifier;
	nIdentifier++;

	// Create a PyCapsule with xla function for XLA compilation
	xlaCapsule = GetXlaCapsule();
	#ifdef HAS_CUDA
		xlaGpuCapsule = GetXlaCapsuleGpu();
	#endif
};

BasisFunc::~BasisFunc(){
	delete[] nC;
};

void BasisFunc::H(double* x, int n, const int d, int* nOut, int* mOut, double** F,  bool full){
	*nOut = n;
	*mOut = full ? m : m-numC;

	int j,k;
	double dMult = pow(c,d);
	double* dark = new double[n*m];
	double* z = new double[n];

	for (k=0;k<n;k++)
		z[k] = (x[k]-x0)*c+z0;

	*F = (double*)malloc((*mOut)*n*sizeof(double));
	Hint(d,z,n,dark);

	if (!full){
		int i=-1;
		bool flag;
		for (j=0;j<m;j++){
			flag = false;
			for (k=0;k<numC;k++){
				if (j == nC[k]){
					flag = true;
					break;
				}
			}
			if (flag) continue; else i++;
			for (k=0;k<n;k++)
				(*F)[(*mOut)*k+i] = dark[m*k+j]*dMult;
		}
	} else {
		for (j=0;j<m;j++){
			for (k=0;k<n;k++)
				(*F)[m*k+j] = dark[m*k+j]*dMult;
		}
	}
	delete[] dark; delete[] z;
};

void BasisFunc::xla(void* out, void** in){
	double* out_buf = reinterpret_cast<double*>(out);
	double* x = reinterpret_cast<double*>(in[1]);
	int d = (reinterpret_cast<int*>(in[2]))[0];
	bool full = (reinterpret_cast<bool*>(in[3]))[0];
	int n = (reinterpret_cast<int*>(in[4]))[0];
	int mOut = (reinterpret_cast<int*>(in[5]))[0];

	int j,k;
	double dMult = pow(c,d);
	double* dark = new double[n*m];
	double* z = new double[n];

	for (k=0;k<n;k++)
		z[k] = (x[k]-x0)*c+z0;
	
	Hint(d,z,n,dark);

	if (!full){
		int i=-1;
		bool flag;
		for (j=0;j<m;j++){
			flag = false;
			for (k=0;k<numC;k++){
				if (j == nC[k]){
					flag = true;
					break;
				}
			}
			if (flag) continue; else i++;
			for (k=0;k<n;k++)
				out_buf[mOut*k+i] = dark[m*k+j]*dMult;
		}
	} else {
		for (j=0;j<m;j++){
			for (k=0;k<n;k++)
				out_buf[m*k+j] = dark[m*k+j]*dMult;
		}
	}
	delete[] dark; delete[] z;
};

#ifdef HAS_CUDA
	void BasisFunc::xlaGpu(CUstream stream, void** buffers, const char* opaque, size_t opaque_len){
		printf("Not implemented yet!\n");
	};
#endif 

PyObject* BasisFunc::GetXlaCapsule(){
	xlaFnType xlaFnPtr = xlaWrapper;
	const char* name = "xla._CUSTOM_CALL_TARGET";
	PyObject* capsule;
	capsule = PyCapsule_New(reinterpret_cast<void*>(xlaFnPtr), name, NULL);
	return capsule;
};

#ifdef HAS_CUDA
	PyObject* BasisFunc::GetXlaCapsuleGpu(){
		xlaGpuFnType xlaFnPtr = xlaGpuWrapper;
		const char* name = "xla._CUSTOM_CALL_TARGET";
		PyObject* capsule;
		capsule = PyCapsule_New(reinterpret_cast<void*>(xlaFnPtr), name, NULL);
		return capsule;
	};
#endif

// COP: **********************************************************************
void CP::Hint(const int d, const double* x, const int nOut, double* dark){

	int j,k;
	int deg = m-1;
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
		RecurseDeriv(d,0,x,nOut,dark,m);
	}
	return;
};

void CP::RecurseDeriv(const int d, int dCurr, const double* x, const int nOut, double*& F, const int mOut){
	if (dCurr != d){
		int j, k;
		double* dark = new double[mOut*nOut];
		memcpy(&dark[0],F,mOut*nOut*sizeof(double));
		if (dCurr == 0){
			for (k=0;k<nOut;k++){
				F[mOut*k] = 0.;
				F[mOut*k+1] = 1.;
			}
		} else if (dCurr == 1){
			for (k=0;k<nOut;k++){
				F[mOut*k+1] = 0.;
			}
		}
		for (k=2;k<mOut;k++){
			for (j=0;j<nOut;j++)
				F[mOut*j+k] = (2.+2.*dCurr)*dark[mOut*j+k-1]+2.*x[j]*F[mOut*j+k-1]-F[mOut*j+k-2];
		}
		dCurr++;
		delete[] dark;
		RecurseDeriv(d,dCurr,x,nOut,F,mOut);
	}
	return;
};

// LeP: **********************************************************************
void LeP::Hint(const int d, const double* x, const int nOut, double* dark){

	int j,k;
	int deg = m-1;
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
		for (k=1;k<m-1;k++){
			for (j=0;j<nOut;j++)
				dark[m*j+k+1] = ((2.*k+1.)*x[j]*dark[m*j+k]-k*dark[m*j+k-1])/(k+1.);
		}
		RecurseDeriv(d,0,x,nOut,dark,m);
	}
	return;
};

void LeP::RecurseDeriv(const int d, int dCurr, const double* x, const int nOut, double*& F, const int mOut){

	if (dCurr != d){
		int j, k;
		double* dark = new double[mOut*nOut];
		memcpy(&dark[0],F,mOut*nOut*sizeof(double));
		if (dCurr == 0){
			for (k=0;k<nOut;k++){
				F[mOut*k] = 0.;
				F[mOut*k+1] = 1.;
			}
		} else if (dCurr == 1){
			for (k=0;k<nOut;k++){
				F[mOut*k+1] = 0.;
			}
		}
		for (k=1;k<mOut-1;k++){
			for (j=0;j<nOut;j++)
				F[m*j+k+1] = ((2.*k+1.)*((dCurr+1.)*dark[m*j+k]+x[j]*F[m*j+k])-k*F[m*j+k-1])/(k+1.);
		}
		dCurr++;
		delete[] dark;
		RecurseDeriv(d,dCurr,x,nOut,F,mOut);
	}
	return;
};

// LaP: **********************************************************************
void LaP::Hint(const int d, const double* x, const int nOut, double* dark){

	int j,k;
	int deg = m-1;
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
				dark[m*k+1] = -1.;
			}
		} else {
			for (k=0;k<nOut;k++){
				dark[m*k] = 1.;
				dark[m*k+1] = 1.-x[k];
			}
		}
	} else {
		for (k=0;k<nOut;k++){
			dark[m*k] = 1.;
			dark[m*k+1] = 1.-x[k];
		}
		for (k=1;k<m-1;k++){
			for (j=0;j<nOut;j++)
				dark[m*j+k+1] = ((2.*k+1.-x[j])*dark[m*j+k]-k*dark[m*j+k-1])/(k+1.);
		}
		RecurseDeriv(d,0,x,nOut,dark,m);
	}
	return;
};

void LaP::RecurseDeriv(const int d, int dCurr, const double* x, const int nOut, double*& F, const int mOut){

	if (dCurr != d){
		int j, k;
		double* dark = new double[mOut*nOut];
		memcpy(&dark[0],F,mOut*nOut*sizeof(double));
		if (dCurr == 0){
			for (k=0;k<nOut;k++){
				F[mOut*k] = 0.;
				F[mOut*k+1] = -1.;
			}
		} else if (dCurr == 1){
			for (k=0;k<nOut;k++){
				F[mOut*k+1] = 0.;
			}
		}
		for (k=1;k<mOut-1;k++){
			for (j=0;j<nOut;j++)
				F[m*j+k+1] = ((2.*k+1.-x[j])*F[m*j+k]-(dCurr+1.)*dark[m*j+k]-k*F[m*j+k-1])/(k+1.);
		}
		dCurr++;
		delete[] dark;
		RecurseDeriv(d,dCurr,x,nOut,F,mOut);
	}
	return;
};

// HoPpro: **********************************************************************
// Hermite polynomials, probablists
void HoPpro::Hint(const int d, const double* x, const int nOut, double* dark){

	int j,k;
	int deg = m-1;
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
				dark[m*k] = 1.;
				dark[m*k+1] = x[k];
			}
		}
	} else {
		for (k=0;k<nOut;k++){
			dark[m*k] = 1.;
			dark[m*k+1] = x[k];
		}
		for (k=1;k<m-1;k++){
			for (j=0;j<nOut;j++)
				dark[m*j+k+1] = x[j]*dark[m*j+k]-k*dark[m*j+k-1];
		}
		RecurseDeriv(d,0,x,nOut,dark,m);
	}
	return;
};

void HoPpro::RecurseDeriv(const int d, int dCurr, const double* x, const int nOut, double*& F, const int mOut){

	if (dCurr != d){
		int j, k;
		double* dark = new double[mOut*nOut];
		memcpy(&dark[0],F,mOut*nOut*sizeof(double));
		if (dCurr == 0){
			for (k=0;k<nOut;k++){
				F[mOut*k] = 0.;
				F[mOut*k+1] = 1.;
			}
		} else if (dCurr == 1){
			for (k=0;k<nOut;k++){
				F[mOut*k+1] = 0.;
			}
		}
		for (k=1;k<mOut-1;k++){
			for (j=0;j<nOut;j++)
				F[m*j+k+1] = (dCurr+1.)*dark[m*j+k]+x[j]*F[m*j+k]-k*F[m*j+k-1];
		}
		dCurr++;
		delete[] dark;
		RecurseDeriv(d,dCurr,x,nOut,F,mOut);
	}
	return;
};

// HoPphy: **********************************************************************
// Hermite polynomials, physicists
void HoPphy::Hint(const int d, const double* x, const int nOut, double* dark){

	int j,k;
	int deg = m-1;
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
				dark[m*k+1] = 2.;
			}
		} else {
			for (k=0;k<nOut;k++){
				dark[m*k] = 1.;
				dark[m*k+1] = 2.*x[k];
			}
		}
	} else {
		for (k=0;k<nOut;k++){
			dark[m*k] = 1.;
			dark[m*k+1] = 2.*x[k];
		}
		for (k=1;k<m-1;k++){
			for (j=0;j<nOut;j++)
				dark[m*j+k+1] = 2.*x[j]*dark[m*j+k]-2.*k*dark[m*j+k-1];
		}
		RecurseDeriv(d,0,x,nOut,dark,m);
	}
	return;
};

void HoPphy::RecurseDeriv(const int d, int dCurr, const double* x, const int nOut, double*& F, const int mOut){

	if (dCurr != d){
		int j, k;
		double* dark = new double[mOut*nOut];
		memcpy(&dark[0],F,mOut*nOut*sizeof(double));
		if (dCurr == 0){
			for (k=0;k<nOut;k++){
				F[mOut*k] = 0.;
				F[mOut*k+1] = 2.;
			}
		} else if (dCurr == 1){
			for (k=0;k<nOut;k++){
				F[mOut*k+1] = 0.;
			}
		}
		for (k=1;k<mOut-1;k++){
			for (j=0;j<nOut;j++)
				F[m*j+k+1] = 2.*(dCurr+1.)*dark[m*j+k]+2.*x[j]*F[m*j+k]-2.*k*F[m*j+k-1];
		}
		dCurr++;
		delete[] dark;
		RecurseDeriv(d,dCurr,x,nOut,F,mOut);
	}
	return;
};

// FS: **********************************************************************
// Fourier Series
void FS::Hint(const int d, const double* x, const int nOut, double* dark){

	int j,k,g;
	if (d == 0){
		for (k=0;k<nOut;k++)
			dark[m*k] = 1;
		for (j=1;j<m;j++){
			for (k=0;k<nOut;k++){
				g = ceil(j/2.);
				if (j%2 == 0){
					dark[m*k+j] = cos(g*x[k]);
				} else {
					dark[m*k+j] = sin(g*x[k]);
				}
			}
		}
	} else {
		for (k=0;k<nOut;k++)
			dark[m*k] = 0;
		if (d%4 == 0){
			for (j=1;j<m;j++){
				for (k=0;k<nOut;k++){
					g = ceil(j/2.);
					if (j%2 == 0){
						dark[m*k+j] = pow(g,d)*cos(g*x[k]);
					} else {
						dark[m*k+j] = pow(g,d)*sin(g*x[k]);
					}
				}
			}
		} else if (d%4 == 1){
			for (j=1;j<m;j++){
				for (k=0;k<nOut;k++){
					g = ceil(j/2.);
					if (j%2 == 0){
						dark[m*k+j] = -pow(g,d)*sin(g*x[k]);
					} else {
						dark[m*k+j] = pow(g,d)*cos(g*x[k]);
					}
				}
			}
		} else if (d%4 == 2){
			for (j=1;j<m;j++){
				for (k=0;k<nOut;k++){
					g = ceil(j/2.);
					if (j%2 == 0){
						dark[m*k+j] = -pow(g,d)*cos(g*x[k]);
					} else {
						dark[m*k+j] = -pow(g,d)*sin(g*x[k]);
					}
				}
			}
		} else {
			for (j=1;j<m;j++){
				for (k=0;k<nOut;k++){
					g = ceil(j/2.);
					if (j%2 == 0){
						dark[m*k+j] = pow(g,d)*sin(g*x[k]);
					} else {
						dark[m*k+j] = -pow(g,d)*cos(g*x[k]);
					}
				}
			}
		}
	}
	return;
};

// ELM: **********************************************************************
// ELM base class
ELM::ELM(double x0, double xf, int* nCin, int ncDim0, int min):
	BasisFunc(x0,xf,nCin,ncDim0,min,0.,1.){

	int k;
	w = new double[m];
	b = new double[m];

	for (k=0;k<m;k++){
		w[k] = 20.*((double)rand()/(double)RAND_MAX)-10.;
		b[k] = 20.*((double)rand()/(double)RAND_MAX)-10.;
	}
};

ELM::~ELM(){
	delete[] b;
	delete[] w;
};

void ELM::setW(double* arrIn, int nIn){
	if (nIn != m){
		printf("Failure in setW function. Weight vector is the wrong size. Exiting program.\n");
		exit(EXIT_FAILURE);
	}
	for (int k=0;k<m;k++)
		w[k] = arrIn[k];
};
	
void ELM::setB(double* arrIn, int nIn){
	if (nIn != m){
		printf("Failure in setB function. Bias vector is the wrong size. Exiting program.\n");
		exit(EXIT_FAILURE);
	}
	for (int k=0;k<m;k++)
		b[k] = arrIn[k];
};

void ELM::getW(double** arrOut, int* nOut){
	*nOut = m;
	*arrOut = (double*)malloc(m*sizeof(double));
	for (int k=0;k<m;k++)
		(*arrOut)[k] = w[k];
	return;
};

void ELM::getB(double** arrOut, int* nOut){
	*nOut = m;
	*arrOut = (double*)malloc(m*sizeof(double));
	for (int k=0;k<m;k++)
		(*arrOut)[k] = b[k];
	return;
};

// ELM ReLU: **********************************************************************
void ELMReLU::Hint(const int d, const double* x, const int nOut, double* dark){
	int j,k;

	if (d == 0){
		for (j=0;j<nOut;j++){
			for (k=0;k<m;k++){
				dark[m*j+k] = std::max(0.,w[k]*x[j]+b[k]);
			}
		}
	} else if (d==1) {
		double dark1;
		for (j=0;j<nOut;j++){
			for (k=0;k<m;k++){
				dark1 = w[k]*x[j]+b[k];
				if (dark1 <= 0.){
					dark[m*j+k] = 0.;
				} else {
					dark[m*j+k] = w[k];
				}
			}
		}
	} else {
		for (j=0;j<nOut;j++){
			for (k=0;k<m;k++){
				dark[m*j+k] = 0.;
			}
		}
	}
	return;
};

// ELM Sigmoid: **********************************************************************
void ELMSigmoid::Hint(const int d, const double* x, const int nOut, double* dark){
	int j,k;

	for (j=0;j<nOut;j++){
		for (k=0;k<m;k++)
			dark[m*j+k] = 1./(1.+exp(-w[k]*x[j]-b[k]));
	}

	switch (d){
		case 0:{
			break;
	    }
		case 1:{
			for (j=0;j<nOut;j++){
				for (k=0;k<m;k++)
					dark[m*j+k] = w[k]*(dark[m*j+k]-pow(dark[m*j+k],2));
			}
			break;
	    }
		case 2:{
			for (j=0;j<nOut;j++){
				for (k=0;k<m;k++)
					dark[m*j+k] = pow(w[k],2)*(dark[m*j+k]-3.*pow(dark[m*j+k],2)+2.*pow(dark[m*j+k],3));
			}
			break;
	    }
		case 3:{
			for (j=0;j<nOut;j++){
				for (k=0;k<m;k++)
					dark[m*j+k] = pow(w[k],3)*(dark[m*j+k]-7.*pow(dark[m*j+k],2)+12.*pow(dark[m*j+k],3)-6.*pow(dark[m*j+k],4));
			}
			break;
	    }
		case 4:{
			for (j=0;j<nOut;j++){
				for (k=0;k<m;k++)
					dark[m*j+k] = pow(w[k],4)*(dark[m*j+k]-15.*pow(dark[m*j+k],2)+50.*pow(dark[m*j+k],3)-60.*pow(dark[m*j+k],4)+24.*pow(dark[m*j+k],5));
			}
			break;
	    }
		case 5:{
			for (j=0;j<nOut;j++){
				for (k=0;k<m;k++)
					dark[m*j+k] = pow(w[k],5)*(dark[m*j+k]-31.*pow(dark[m*j+k],2)+180.*pow(dark[m*j+k],3)-390.*pow(dark[m*j+k],4)+360.*pow(dark[m*j+k],5)-120.*pow(dark[m*j+k],6));
			}
			break;
	    }
		case 6:{
			for (j=0;j<nOut;j++){
				for (k=0;k<m;k++)
					dark[m*j+k] = pow(w[k],6)*(dark[m*j+k]-63.*pow(dark[m*j+k],2)+602.*pow(dark[m*j+k],3)-2100.*pow(dark[m*j+k],4)+3360.*pow(dark[m*j+k],5)-2520.*pow(dark[m*j+k],6)+720.*pow(dark[m*j+k],7));
			}
			break;
	    }
		case 7:{
			for (j=0;j<nOut;j++){
				for (k=0;k<m;k++)
					dark[m*j+k] = pow(w[k],7)*(dark[m*j+k]-127.*pow(dark[m*j+k],2)+1932.*pow(dark[m*j+k],3)-10206.*pow(dark[m*j+k],4)+25200.*pow(dark[m*j+k],5)-31920.*pow(dark[m*j+k],6)+20160.*pow(dark[m*j+k],7)-5040.*pow(dark[m*j+k],8));
			}
			break;
	    }
		case 8:{
			for (j=0;j<nOut;j++){
				for (k=0;k<m;k++)
					dark[m*j+k] = pow(w[k],8)*(dark[m*j+k]-255.*pow(dark[m*j+k],2)+6050.*pow(dark[m*j+k],3)-46620.*pow(dark[m*j+k],4)+166824.*pow(dark[m*j+k],5)-317520.*pow(dark[m*j+k],6)+332640.*pow(dark[m*j+k],7)-181440.*pow(dark[m*j+k],8)+40320.*pow(dark[m*j+k],9));
			}
			break;
	    }
	}
	return;
};

// ELM Tanh: **********************************************************************

void ELMTanh::Hint(const int d, const double* x, const int nOut, double* dark){
	int j,k;

	for (j=0;j<nOut;j++){
		for (k=0;k<m;k++)
			dark[m*j+k] = tanh(w[k]*x[j]+b[k]);
	}

	switch (d){
		case 0:{
			break;
	    }
		case 1:{
			for (j=0;j<nOut;j++){
				for (k=0;k<m;k++)
					dark[m*j+k] = w[k]*(1. - pow(dark[m*j+k],2));
			}
			break;
	    }
		case 2:{
			for (j=0;j<nOut;j++){
				for (k=0;k<m;k++)
					dark[m*j+k] = pow(w[k],2)*(-2. * dark[m*j+k] + 2. * pow(dark[m*j+k],3));
			}
			break;
	    }
		case 3:{
			for (j=0;j<nOut;j++){
				for (k=0;k<m;k++)
					dark[m*j+k] = pow(w[k],3)*(-2. + 8. * pow(dark[m*j+k],2) - 6. * pow(dark[m*j+k],4));
			}
			break;
	    }
		case 4:{
			for (j=0;j<nOut;j++){
				for (k=0;k<m;k++)
					dark[m*j+k] = pow(w[k],4)*(16. * dark[m*j+k] - 40. * pow(dark[m*j+k],3) + 24. * pow(dark[m*j+k],5));
			}
			break;
	    }
		case 5:{
			for (j=0;j<nOut;j++){
				for (k=0;k<m;k++)
					dark[m*j+k] = pow(w[k],5)*(16. - 136. * pow(dark[m*j+k],2) + 240. * pow(dark[m*j+k],4) -120. * pow(dark[m*j+k],6));
			}
			break;
	    }
		case 6:{
			for (j=0;j<nOut;j++){
				for (k=0;k<m;k++)
					dark[m*j+k] = pow(w[k],6)*(-272. * dark[m*j+k] + 1232. * pow(dark[m*j+k],3) - 1680. * pow(dark[m*j+k],5) + 720. * pow(dark[m*j+k],7));
			}
			break;
	    }
		case 7:{
			for (j=0;j<nOut;j++){
				for (k=0;k<m;k++)
					dark[m*j+k] = pow(w[k],7)*(-272. + 3968. * pow(dark[m*j+k],2) - 12096. * pow(dark[m*j+k],4) + 13440. * pow(dark[m*j+k],6) - 5040. * pow(dark[m*j+k],8));
			}
			break;
	    }
		case 8:{
			for (j=0;j<nOut;j++){
				for (k=0;k<m;k++)
					dark[m*j+k] = pow(w[k],8)*(7936. * dark[m*j+k] - 56320. * pow(dark[m*j+k],3) + 129024. * pow(dark[m*j+k],5) - 120960. * pow(dark[m*j+k],7) + 40320. * pow(dark[m*j+k],9));
			}
			break;
	    }
	}
	return;
};

// ELM Sin: **********************************************************************

void ELMSin::Hint(const int d, const double* x, const int nOut, double* dark){
	int j,k;

	if (d%4 == 0){
		for (j=0;j<nOut;j++){
			for (k=0;k<m;k++){
				dark[m*j+k] = pow(w[k],d) * sin(w[k]*x[j]+b[k]);
			}
		}
	} else if (d%4 == 1){
		for (j=0;j<nOut;j++){
			for (k=0;k<m;k++){
				dark[m*j+k] = pow(w[k],d) * cos(w[k]*x[j]+b[k]);
			}
		}
	} else if (d%4 == 2){
		for (j=0;j<nOut;j++){
			for (k=0;k<m;k++){
				dark[m*j+k] = -pow(w[k],d) * sin(w[k]*x[j]+b[k]);
			}
		}
	} else {
		for (j=0;j<nOut;j++){
			for (k=0;k<m;k++){
				dark[m*j+k] = -pow(w[k],d) * cos(w[k]*x[j]+b[k]);
			}
		}

	}
	return;
};

// ELM Swish: **********************************************************************

void ELMSwish::Hint(const int d, const double* x, const int nOut, double* dark){
	int j,k;
	double sig[nOut*m], zint[nOut*m];

	if (d == 0){
		for (j=0;j<nOut;j++){
			for (k=0;k<m;k++)
				dark[m*j+k] =  (w[k]*x[j]+b[k]) * 1./(1.+exp(-w[k]*x[j]-b[k]));
		}

	} else {
		for (j=0;j<nOut;j++){
			for (k=0;k<m;k++){
				zint[m*j+k] = w[k]*x[j]+b[k];
				sig[m*j+k] = 1./(1.+exp(-zint[m*j+k]));
			}
		}
		switch (d){
			case 1:{
				for (j=0;j<nOut;j++){
					for (k=0;k<m;k++)
						dark[m*j+k] = w[k]*(sig[m*j+k] + zint[m*j+k] * ( sig[m*j+k]-pow(sig[m*j+k],2) ));
				}
				break;
		    }
			case 2:{
				for (j=0;j<nOut;j++){
					for (k=0;k<m;k++)
						dark[m*j+k] = pow(w[k],2) * (2.*(sig[m*j+k]-pow(sig[m*j+k],2)) + zint[m*j+k] * ( sig[m*j+k]-3.*pow(sig[m*j+k],2)+2.*pow(sig[m*j+k],3) ));
				}
				break;
		    }
			case 3:{
				for (j=0;j<nOut;j++){
					for (k=0;k<m;k++)
						dark[m*j+k] = pow(w[k],3)*( 3.*( sig[m*j+k]-3.*pow(sig[m*j+k],2)+2.*pow(sig[m*j+k],3) ) + zint[m*j+k] * (sig[m*j+k]-7.*pow(sig[m*j+k],2)+12.*pow(sig[m*j+k],3)-6.*pow(sig[m*j+k],4)));
				}
				break;
		    }
			case 4:{
				for (j=0;j<nOut;j++){
					for (k=0;k<m;k++)
						dark[m*j+k] = pow(w[k],4)*( 4.*( sig[m*j+k]-7.*pow(sig[m*j+k],2)+12.*pow(sig[m*j+k],3)-6.*pow(sig[m*j+k],4) ) + zint[m*j+k] * ( sig[m*j+k]-15.*pow(sig[m*j+k],2)+50.*pow(sig[m*j+k],3)-60.*pow(sig[m*j+k],4)+24.*pow(sig[m*j+k],5) ));
				}
				break;
		    }
			case 5:{
				for (j=0;j<nOut;j++){
					for (k=0;k<m;k++)

						dark[m*j+k] = pow(w[k],5)*( 5.*( sig[m*j+k]-15.*pow(sig[m*j+k],2)+50.*pow(sig[m*j+k],3)-60.*pow(sig[m*j+k],4)+24.*pow(sig[m*j+k],5) ) + zint[m*j+k] * ( sig[m*j+k]-31.*pow(sig[m*j+k],2)+180.*pow(sig[m*j+k],3)-390.*pow(sig[m*j+k],4)+360.*pow(sig[m*j+k],5)-120.*pow(sig[m*j+k],6) ));
				}
				break;
		    }
			case 6:{
				for (j=0;j<nOut;j++){
					for (k=0;k<m;k++)
						dark[m*j+k] = pow(w[k],6) * ( 6.*( sig[m*j+k]-31.*pow(sig[m*j+k],2)+180.*pow(sig[m*j+k],3)-390.*pow(sig[m*j+k],4)+360.*pow(sig[m*j+k],5)-120.*pow(sig[m*j+k],6) ) + zint[m*j+k] * ( sig[m*j+k]-63.*pow(sig[m*j+k],2)+602.*pow(sig[m*j+k],3)-2100.*pow(sig[m*j+k],4)+3360.*pow(sig[m*j+k],5)-2520.*pow(sig[m*j+k],6)+720.*pow(sig[m*j+k],7) ));
				}
				break;
		    }
			case 7:{
				for (j=0;j<nOut;j++){
					for (k=0;k<m;k++)
						dark[m*j+k] = pow(w[k],7)*( 7.*( sig[m*j+k]-63.*pow(sig[m*j+k],2)+602.*pow(sig[m*j+k],3)-2100.*pow(sig[m*j+k],4)+3360.*pow(sig[m*j+k],5)-2520.*pow(sig[m*j+k],6)+720.*pow(sig[m*j+k],7) ) + zint[m*j+k] * ( sig[m*j+k]-127.*pow(sig[m*j+k],2)+1932.*pow(sig[m*j+k],3)-10206.*pow(sig[m*j+k],4)+25200.*pow(sig[m*j+k],5)-31920.*pow(sig[m*j+k],6)+20160.*pow(sig[m*j+k],7)-5040.*pow(sig[m*j+k],8) ));
				}
				break;
		    }
			case 8:{
				for (j=0;j<nOut;j++){
					for (k=0;k<m;k++)
						dark[m*j+k] = pow(w[k],8)*( 8.*( sig[m*j+k]-127.*pow(sig[m*j+k],2)+1932.*pow(sig[m*j+k],3)-10206.*pow(sig[m*j+k],4)+25200.*pow(sig[m*j+k],5)-31920.*pow(sig[m*j+k],6)+20160.*pow(sig[m*j+k],7)-5040.*pow(sig[m*j+k],8) ) + zint[m*j+k] * ( sig[m*j+k]-255.*pow(sig[m*j+k],2)+6050.*pow(sig[m*j+k],3)-46620.*pow(sig[m*j+k],4)+166824.*pow(sig[m*j+k],5)-317520.*pow(sig[m*j+k],6)+332640.*pow(sig[m*j+k],7)-181440.*pow(sig[m*j+k],8)+40320.*pow(sig[m*j+k],9) ));
				}
				break;
		    }
		}
	}
	return;
};

// Parent n-dimensional basis function class: **********************************************************************
nBasisFunc::nBasisFunc(double* x0in, int x0Dim0, double* xf, int xfDim0, int* nCin, int ncDim0, int ncDim1, int min, double z0in, double zfin){

	// Initialize internal variables based on user givens
	dim = x0Dim0;
	m = min;
	numC = ncDim1;
	z0 = z0in;
	zf = zfin;

	x0 = new double[dim];
	memcpy(x0,x0in,dim*sizeof(double));

	nC = new int[dim*numC];
	memcpy(nC,nCin,ncDim0*ncDim1*sizeof(int));

	c = new double[dim];
	for (int k=0; k<dim; k++)
		c[k] = (zf-z0)/(xf[k]-x0[k]);

	// Calculate the number of basis functions
	int vec[dim];
	numBasisFunc = 0; numBasisFuncFull = 0;
	NumBasisFunc(dim-1, &vec[0], numBasisFunc, false);
	NumBasisFunc(dim-1, &vec[0], numBasisFuncFull, true);

	// Track this instance of BasisFunc 
	BasisFuncContainer.push_back(this);
	identifier = nIdentifier;
	nIdentifier++;

	// Create a PyCapsule with xla function for XLA compilation
	xlaCapsule = GetXlaCapsule();
	#ifdef HAS_CUDA
		xlaGpuCapsule = GetXlaCapsuleGpu();
	#endif
}

nBasisFunc::~nBasisFunc(){
	delete[] c;
};

void nBasisFunc::getC(double** arrOut, int* nOut){
	*nOut = dim;
	*arrOut = (double*)malloc(dim*sizeof(double));
	for (int k=0;k<dim;k++)
		(*arrOut)[k] = c[k];
	return;
};

void nBasisFunc::H(double* x, int in, int xDim1, int* d, int dDim0, int* nOut, int* mOut, double** F, const bool full){
	int numBasis = full ? numBasisFuncFull : numBasisFunc;
	*mOut = numBasis;
	*nOut = xDim1;
	*F = (double*)malloc(numBasis*xDim1*sizeof(double));
	nHint(x,xDim1,d,dDim0,numBasis,*F,full);
};

void nBasisFunc::xla(void* out, void** in){
	double* out_buf = reinterpret_cast<double*>(out);
	double* x = reinterpret_cast<double*>(in[1]);
	int* d = reinterpret_cast<int*>(in[2]);
	int dDim0 = (reinterpret_cast<int*>(in[3]))[0];
	bool full = (reinterpret_cast<bool*>(in[4]))[0];
	int nOut = (reinterpret_cast<int*>(in[5]))[0];
	int mOut = (reinterpret_cast<int*>(in[6]))[0];

	nHint(x,nOut,d,dDim0,mOut,out_buf,full);

};

void nBasisFunc::nHint(double* x, int n, const int* d, int dDim0, int numBasis, double*& F, const bool full){

	int j,k;
	double* dark = new double[n*m]; 
	double* T = new double[n*m*dim]; 
	double* z = new double[n*dim];
	double dMult;

	// Calculate the basis function domain points
	for (j=0;j<dim;j++){
		for (k=0;k<n;k++)
			z[j*n+k] = (x[j*n+k]-x0[j])*c[j]+z0;
	}

	// Calculate univariate basis functions
	for (k=0;k<dim;k++){
		if (k >= dDim0){
			Hint(0,z+k*n,n,dark);
			dMult = 1.;
		} else {
			Hint(d[k],z+k*n,n,dark);
			dMult = pow(c[k],d[k]);
		}
		for (j=0;j<n*m;j++)
			T[j+k*m*n] = dark[j]*dMult;
	}

	for (k=0;k<n*numBasis;k++)
		F[k] = 1.;

	int count = 0;
	int vec[dim];
	RecurseBasis(dim-1, &vec[0], count, full, n, numBasis, &T[0], F);
	delete[] dark; delete[] T; delete[] z;
};

void nBasisFunc::NumBasisFunc(int dimCurr, int* vec, int &count, const bool full){
	int k;
	if (dimCurr > 0){
		for (k=0;k<m;k++){
			vec[dimCurr] = k;
			NumBasisFunc(dimCurr-1,vec,count,full);
		}
	} else {
		int j,g;
		int sum;
		bool flag, flag1;
		for (k=0;k<m;k++){
			vec[dimCurr] = k;
			flag = false;
			sum = 0;
			if (full){
				for (j=0;j<dim;j++)
					sum += vec[j];
				if (sum <= m-1)
					count ++;
			} else {

				// If at least one of the dimensions' basis functions is not a constraint, then
				// set flag = true
				for (j=0;j<dim;j++){
					flag1 = true;
					for (g=0;g<numC;g++){
						if (vec[j] == nC[j*numC+g]){
							flag1 = false;
							break;
						}
					}
					if (flag1) flag = true;
				}

				// If flag is true and the degree of the product of univariate basis
				// functions is less than the degree specified, add one to count
				if (flag){
					for (j=0;j<dim;j++)
						sum += vec[j];
					if (sum <= m-1)
						count ++;
				}
			}
		}
	}
	return;
};

void nBasisFunc::RecurseBasis(int dimCurr, int* vec, int &count, const bool full, const int in, const int numBasis, const double* T, double* out){
	int k;
	if (dimCurr > 0){
		for (k=0;k<m;k++){
			vec[dimCurr] = k;
			RecurseBasis(dimCurr-1,vec,count,full,in,numBasis,T,out);
		}
	} else {
		int j,g,h,l;
		int sum;
		bool flag, flag1;
		for (k=0;k<m;k++){
			vec[dimCurr] = k;
			flag = false;
			sum = 0;
			if (full){
				for (j=0;j<dim;j++)
					sum += vec[j];
				if (sum <= m-1){
					for (h=0;h<in;h++){
						for (l=0;l<dim;l++)
							out[h*numBasis+count] *= T[m*in*l+vec[l]+h*m];
					}
					count ++;
				}
			} else {

				// If at least one of the dimensions' basis functions is not a constraint, then
				// set flag = true
				for (j=0;j<dim;j++){
					flag1 = true;
					for (g=0;g<numC;g++){
						if (vec[j] == nC[j*numC+g]){
							flag1 = false;
							break;
						}
					}
					if (flag1) flag = true;
				}

				// If flag is true and the degree of the product of univariate basis
				// functions is less than the degree specified, add one to count
				if (flag){
					for (j=0;j<dim;j++)
						sum += vec[j];
					if (sum <= m-1){
						for (h=0;h<in;h++){
							for (l=0;l<dim;l++)
								out[h*numBasis+count] *= T[m*in*l+vec[l]+h*m];
						}
						count ++;
					}
				}
			}
		}
	}
	return;
};

// nELM base class: ***********************************************************************************
nELM::nELM(double* x0in, int x0Dim0, double* xf, int xfDim0, int* nCin, int ncDim0, int min, double z0in, double zfin){

	int k;
	bool flag = true;

	// Initialize internal variables based on user givens
	dim = x0Dim0;
	m = min;
	numC = ncDim0;
	z0 = z0in;
	zf = zfin;

	x0 = new double[dim];
	memcpy(x0,x0in,dim*sizeof(double));

	nC = new int[dim*numC];
	memcpy(nC,nCin,ncDim0*sizeof(int));

	c = new double[dim];
	for (k=0; k<dim; k++)
		c[k] = (zf-z0)/(xf[k]-x0[k]);

	for (k=0;k<ncDim0;k++){
		if (nC[k] != -1){
			flag = false;
		}
	}
	if (flag) numC = 0;

	m = min;

	// Calculate the number of basis functions
	numBasisFunc = m-numC;
   	numBasisFuncFull = m;

	// Track this instance of BasisFunc 
	BasisFuncContainer.push_back(this);
	identifier = nIdentifier;
	nIdentifier++;

	// Create a PyCapsule with xla function for XLA compilation
	xlaCapsule = GetXlaCapsule();
	#ifdef HAS_CUDA
		xlaGpuCapsule = GetXlaCapsuleGpu();
	#endif
		
	w = new double[dim*m];
	b = new double[m];
	for (k=0;k<dim*m;k++)
		w[k] = 2.*((double)rand()/(double)RAND_MAX)-1.;
	for (k=0;k<m;k++)
		b[k] = 2.*((double)rand()/(double)RAND_MAX)-1.;

};

nELM::~nELM(){
	delete[] b;
	delete[] w;
};

void nELM::setW(double* arrIn, int dimIn, int nIn){
	if ((nIn != m)||(dimIn != dim)){
		printf("Failure in setW function. Weight vector is the wrong size. Exiting program.\n");
		exit(EXIT_FAILURE);
	}
	for (int k=0;k<m*dim;k++)
		w[k] = arrIn[k];
};
	
void nELM::setB(double* arrIn, int nIn){
	if (nIn != m){
		printf("Failure in setB function. Bias vector is the wrong size. Exiting program.\n");
		exit(EXIT_FAILURE);
	}
	for (int k=0;k<m;k++)
		b[k] = arrIn[k];
};

void nELM::getW(int* dimOut, int* nOut, double** arrOut){
	*dimOut = dim;
	*nOut = m;
	*arrOut = (double*)malloc(m*dim*sizeof(double));
	for (int k=0;k<m*dim;k++)
		(*arrOut)[k] = w[k];
	return;
};

void nELM::getB(double** arrOut, int* nOut){
	*nOut = m;
	*arrOut = (double*)malloc(m*sizeof(double));
	for (int k=0;k<m;k++)
		(*arrOut)[k] = b[k];
	return;
};

void nELM::nHint(double* x, int n, const int* d, int dDim0, int numBasis, double*& F, const bool full){

	int j,k;
	double* z = new double[n*dim];

	// Calculate the basis function domain points
	for (j=0;j<dim;j++){
		for (k=0;k<n;k++)
			z[j*n+k] = (x[j*n+k]-x0[j])*c[j]+z0;
	}

	if ((numC == 0) || (full)){
		nElmHint(d,dDim0,z,n,F);
	} else {
		int i=-1;
		bool flag;
		double* dark = new double[m*n]; // Allocated on the heap, as allocating on the stack frequently causes segfaults due to size. - CL
		nElmHint(d,dDim0,z,n,dark);

		for (j=0;j<m;j++){
			flag = false;
			for (k=0;k<numC;k++){
				if (j == nC[k]){
					flag = true;
					break;
				}
			}
			if (flag) continue; else i++;
			for (k=0;k<n;k++)
				F[numBasis*k+i] = dark[m*k+j];
		}

		delete[] dark;
	}
	delete[] z;

};

// nELM Sigmoid *******************************************************************************************
void nELMSigmoid::nElmHint(const int* d, int dDim0, const double* x, const int in, double* F){

	int i,j,k;
	double dark,dark2=1.;
	int dark1=0;
	for (j=0;j<in;j++){
		for (k=0;k<m;k++){
			dark = 0.;
			for (i=0;i<dim;i++)
				dark += w[i*m+k]*x[i*in+j];
			F[m*j+k] = 1./(1.+exp(-dark-b[k]));
		}
	}

	for (i=0;i<dDim0;i++){
		dark1 += d[i];
		dark2 *= pow(c[i],d[i]);
	}

	switch (dark1){
		case 0:{
			break;
	    }
		case 1:{
			for (j=0;j<in;j++){
				for (k=0;k<m;k++){
					dark = 1.;
					for (i=0;i<dDim0;i++)
						dark *= pow(w[i*m+k],d[i]);
					F[m*j+k] = dark2*dark*(F[m*j+k]-pow(F[m*j+k],2));
				}
			}
			break;
	    }
		case 2:{
			for (j=0;j<in;j++){
				for (k=0;k<m;k++){
					dark = 1.;
					for (i=0;i<dDim0;i++)
						dark *= pow(w[i*m+k],d[i]);
					F[m*j+k] = dark2*dark*(F[m*j+k]-3.*pow(F[m*j+k],2)+2.*pow(F[m*j+k],3));
				}
			}
			break;
	    }
		case 3:{
			for (j=0;j<in;j++){
				for (k=0;k<m;k++){
					dark = 1.;
					for (i=0;i<dDim0;i++)
						dark *= pow(w[i*m+k],d[i]);
					F[m*j+k] = dark2*dark*(F[m*j+k]-7.*pow(F[m*j+k],2)+12.*pow(F[m*j+k],3)-6.*pow(F[m*j+k],4));
				}
			}
			break;
	    }
		case 4:{
			for (j=0;j<in;j++){
				for (k=0;k<m;k++){
					dark = 1.;
					for (i=0;i<dDim0;i++)
						dark *= pow(w[i*m+k],d[i]);
					F[m*j+k] = dark2*dark*(F[m*j+k]-15.*pow(F[m*j+k],2)+50.*pow(F[m*j+k],3)-60.*pow(F[m*j+k],4)+24.*pow(F[m*j+k],5));
				}
			}
			break;
	    }
		case 5:{
			for (j=0;j<in;j++){
				for (k=0;k<m;k++){
					dark = 1.;
					for (i=0;i<dDim0;i++)
						dark *= pow(w[i*m+k],d[i]);
					F[m*j+k] = dark2*dark*(F[m*j+k]-31.*pow(F[m*j+k],2)+180.*pow(F[m*j+k],3)-390.*pow(F[m*j+k],4)+360.*pow(F[m*j+k],5)-120.*pow(F[m*j+k],6));
				}
			}
			break;
	    }
		case 6:{
			for (j=0;j<in;j++){
				for (k=0;k<m;k++){
					dark = 1.;
					for (i=0;i<dDim0;i++)
						dark *= pow(w[i*m+k],d[i]);
					F[m*j+k] = dark2*dark*(F[m*j+k]-63.*pow(F[m*j+k],2)+602.*pow(F[m*j+k],3)-2100.*pow(F[m*j+k],4)+3360.*pow(F[m*j+k],5)-2520.*pow(F[m*j+k],6)+720.*pow(F[m*j+k],7));
				}
			}
			break;
	    }
		case 7:{
			for (j=0;j<in;j++){
				for (k=0;k<m;k++){
					dark = 1.;
					for (i=0;i<dDim0;i++)
						dark *= pow(w[i*m+k],d[i]);
					F[m*j+k] = dark2*dark*(F[m*j+k]-127.*pow(F[m*j+k],2)+1932.*pow(F[m*j+k],3)-10206.*pow(F[m*j+k],4)+25200.*pow(F[m*j+k],5)-31920.*pow(F[m*j+k],6)+20160.*pow(F[m*j+k],7)-5040.*pow(F[m*j+k],8));
				}
			}
			break;
	    }
		case 8:{
			for (j=0;j<in;j++){
				for (k=0;k<m;k++){
					dark = 1.;
					for (i=0;i<dDim0;i++)
						dark *= pow(w[i*m+k],d[i]);
					F[m*j+k] = dark2*dark*(F[m*j+k]-255.*pow(F[m*j+k],2)+6050.*pow(F[m*j+k],3)-46620.*pow(F[m*j+k],4)+166824.*pow(F[m*j+k],5)-317520.*pow(F[m*j+k],6)+332640.*pow(F[m*j+k],7)-181440.*pow(F[m*j+k],8)+40320.*pow(F[m*j+k],9));
				}
			}
			break;
	    }
	}
	return;
};

// nELM Tanh *******************************************************************************************
void nELMTanh::nElmHint(const int* d, int dDim0, const double* x, const int in, double* F){

	int i,j,k;
	double dark,dark2=1.;
	int dark1=0;
	for (j=0;j<in;j++){
		for (k=0;k<m;k++){
			dark = 0.;
			for (i=0;i<dim;i++)
				dark += w[i*m+k]*x[i*in+j];
			F[m*j+k] = tanh(dark+b[k]);

		}
	}
	for (i=0;i<dDim0;i++){
		dark1 += d[i];
		dark2 *= pow(c[i],d[i]);
	}

	switch (dark1){
		case 0:{
			break;
	    }
		case 1:{
			for (j=0;j<in;j++){
				for (k=0;k<m;k++){
					dark = 1.;
					for (i=0;i<dDim0;i++)
						dark *= pow(w[i*m+k],d[i]);
					F[m*j+k] = dark2*dark*(1. - pow(F[m*j+k],2));
				}
			}
			break;
	    }
		case 2:{
			for (j=0;j<in;j++){
				for (k=0;k<m;k++){
					dark = 1.;
					for (i=0;i<dDim0;i++)
						dark *= pow(w[i*m+k],d[i]);
					F[m*j+k] = dark2*dark*(-2. * F[m*j+k] + 2. * pow(F[m*j+k],3));
				}
			}
			break;
	    }
		case 3:{
			for (j=0;j<in;j++){
				for (k=0;k<m;k++){
					dark = 1.;
					for (i=0;i<dDim0;i++)
						dark *= pow(w[i*m+k],d[i]);
					F[m*j+k] = dark2*dark*(-2. + 8. * pow(F[m*j+k],2) - 6. * pow(F[m*j+k],4));
				}
			}
			break;
	    }
		case 4:{
			for (j=0;j<in;j++){
				for (k=0;k<m;k++){
					dark = 1.;
					for (i=0;i<dDim0;i++)
						dark *= pow(w[i*m+k],d[i]);
					F[m*j+k] = dark2*dark*(16. * F[m*j+k] - 40. * pow(F[m*j+k],3) + 24. * pow(F[m*j+k],5));
				}
			}
			break;
	    }
		case 5:{
			for (j=0;j<in;j++){
				for (k=0;k<m;k++){
					dark = 1.;
					for (i=0;i<dDim0;i++)
						dark *= pow(w[i*m+k],d[i]);
					F[m*j+k] = dark2*dark*(16. - 136. * pow(F[m*j+k],2) + 240. * pow(F[m*j+k],4) -120. * pow(F[m*j+k],6));
				}
			}
			break;
	    }
		case 6:{
			for (j=0;j<in;j++){
				for (k=0;k<m;k++){
					dark = 1.;
					for (i=0;i<dDim0;i++)
						dark *= pow(w[i*m+k],d[i]);
					F[m*j+k] = dark2*dark*(-272. * F[m*j+k] + 1232. * pow(F[m*j+k],3) - 1680. * pow(F[m*j+k],5) + 720. * pow(F[m*j+k],7));
				}
			}
			break;
	    }
		case 7:{
			for (j=0;j<in;j++){
				for (k=0;k<m;k++){
					dark = 1.;
					for (i=0;i<dDim0;i++)
						dark *= pow(w[i*m+k],d[i]);
					F[m*j+k] = dark2*dark*(-272. + 3968. * pow(F[m*j+k],2) - 12096. * pow(F[m*j+k],4) + 13440. * pow(F[m*j+k],6) - 5040. * pow(F[m*j+k],8));
				}
			}
			break;
	    }
		case 8:{
			for (j=0;j<in;j++){
				for (k=0;k<m;k++){
					dark = 1.;
					for (i=0;i<dDim0;i++)
						dark *= pow(w[i*m+k],d[i]);
					F[m*j+k] = dark2*dark*(7936. * F[m*j+k] - 56320. * pow(F[m*j+k],3) + 129024. * pow(F[m*j+k],5) - 120960. * pow(F[m*j+k],7) + 40320. * pow(F[m*j+k],9));
				}
			}
			break;
	    }
	}
	return;
};

// nELM Sin *******************************************************************************************
void nELMSin::nElmHint(const int* d, int dDim0, const double* x, const int in, double* F){

	int i,j,k;
	double dark, darkw, dark2=1.;
	int dark1=0;

	for (i=0;i<dDim0;i++){
		dark1 += d[i];
		dark2 *= pow(c[i],d[i]);
	}

	if (dark1%4 == 0){
		for (j=0;j<in;j++){
			for (k=0;k<m;k++){
				dark = 1.; darkw = 0.;
				for (i=0;i<dDim0;i++)
					dark *= pow(w[i*m+k],d[i]);
				for (i=0;i<dim;i++)
					darkw += w[i*m+k] * x[i*in+j];
				F[m*j+k] = dark2 * dark * sin(darkw + b[k]);
			}
		}
	} else if (dark1%4 == 1){
		for (j=0;j<in;j++){
			for (k=0;k<m;k++){
				dark = 1.; darkw = 0.;
				for (i=0;i<dDim0;i++)
					dark *= pow(w[i*m+k],d[i]);
				for (i=0;i<dim;i++)
					darkw += w[i*m+k] * x[i*in+j];
				F[m*j+k] = dark2 * dark * cos(darkw + b[k]);
			}
		}
	} else if (dark1%4 == 2){
		for (j=0;j<in;j++){
			for (k=0;k<m;k++){
				dark = 1.; darkw = 0.;
				for (i=0;i<dDim0;i++)
					dark *= pow(w[i*m+k],d[i]);
				for (i=0;i<dim;i++)
					darkw += w[i*m+k] * x[i*in+j];
				F[m*j+k] = - dark2 * dark * sin(darkw + b[k]);
			}
		}
	} else {
		for (j=0;j<in;j++){
			for (k=0;k<m;k++){
				dark = 1.; darkw = 0.;
				for (i=0;i<dDim0;i++)
					dark *= pow(w[i*m+k],d[i]);
				for (i=0;i<dim;i++)
					darkw += w[i*m+k] * x[i*in+j];
				F[m*j+k] = - dark2 * dark * cos(darkw + b[k]);
			}
		}
	}
	return;
};

// nELM Swish *******************************************************************************************
void nELMSwish::nElmHint(const int* d, int dDim0, const double* x, const int in, double* F){

	int i,j,k;
	int dark1=0;
	double dark,dark2=1.;
	double* sig = new double[in*m]; // Allocated on the heap, as allocating on the stack frequently causes segfaults due to size. - CL
	double* zint = new double[in*m]; // Allocated on the heap, as allocating on the stack frequently causes segfaults due to size. - CL
	for (i=0;i<dDim0;i++){
		dark1 += d[i];
		dark2 *= pow(c[i],d[i]);
	}

	if (dark1 == 0){
		for (j=0;j<in;j++){
			for (k=0;k<m;k++){
				dark = 0.;
				for (i=0;i<dim;i++)
					dark += w[i*m+k]*x[i*in+j];
				F[m*j+k] = (dark+b[k]) * 1./(1.+exp(-dark-b[k]));
			}
		}

	} else {
		for (j=0;j<in;j++){
			for (k=0;k<m;k++){
				dark = 0.;
				for (i=0;i<dim;i++)
					dark += w[i*m+k]*x[i*in+j];
				zint[m*j+k] = dark + b[k];
				sig[m*j+k] = 1./(1.+exp(-zint[m*j+k]));
			}
		}

		switch (dark1){
			case 1:{
				for (j=0;j<in;j++){
					for (k=0;k<m;k++){
						dark = 1.;
						for (i=0;i<dDim0;i++)
							dark *= pow(w[i*m+k],d[i]);
						F[m*j+k] = dark2*dark*(sig[m*j+k] + zint[m*j+k] * ( sig[m*j+k]-pow(sig[m*j+k],2) ));
					}
				}
				break;
		    }
			case 2:{
				for (j=0;j<in;j++){
					for (k=0;k<m;k++){
						dark = 1.;
						for (i=0;i<dDim0;i++)
							dark *= pow(w[i*m+k],d[i]);
						F[m*j+k] = dark2*dark*(2.*(sig[m*j+k]-pow(sig[m*j+k],2)) + zint[m*j+k] * ( sig[m*j+k]-3.*pow(sig[m*j+k],2)+2.*pow(sig[m*j+k],3) ));
					}
				}
				break;
		    }
			case 3:{
				for (j=0;j<in;j++){
					for (k=0;k<m;k++){
						dark = 1.;
						for (i=0;i<dDim0;i++)
							dark *= pow(w[i*m+k],d[i]);
						F[m*j+k] = dark2*dark*( 3.*( sig[m*j+k]-3.*pow(sig[m*j+k],2)+2.*pow(sig[m*j+k],3) ) + zint[m*j+k] * (sig[m*j+k]-7.*pow(sig[m*j+k],2)+12.*pow(sig[m*j+k],3)-6.*pow(sig[m*j+k],4)));
					}
				}
				break;
		    }
			case 4:{
				for (j=0;j<in;j++){
					for (k=0;k<m;k++){
						dark = 1.;
						for (i=0;i<dDim0;i++)
							dark *= pow(w[i*m+k],d[i]);
						F[m*j+k] = dark2*dark*( 4.*( sig[m*j+k]-7.*pow(sig[m*j+k],2)+12.*pow(sig[m*j+k],3)-6.*pow(sig[m*j+k],4) ) + zint[m*j+k] * ( sig[m*j+k]-15.*pow(sig[m*j+k],2)+50.*pow(sig[m*j+k],3)-60.*pow(sig[m*j+k],4)+24.*pow(sig[m*j+k],5) ));
					}
				}
				break;
		    }
			case 5:{
				for (j=0;j<in;j++){
					for (k=0;k<m;k++){
						dark = 1.;
						for (i=0;i<dDim0;i++)
							dark *= pow(w[i*m+k],d[i]);
						F[m*j+k] = dark2*dark*( 5.*( sig[m*j+k]-15.*pow(sig[m*j+k],2)+50.*pow(sig[m*j+k],3)-60.*pow(sig[m*j+k],4)+24.*pow(sig[m*j+k],5) ) + zint[m*j+k] * ( sig[m*j+k]-31.*pow(sig[m*j+k],2)+180.*pow(sig[m*j+k],3)-390.*pow(sig[m*j+k],4)+360.*pow(sig[m*j+k],5)-120.*pow(sig[m*j+k],6) ));
					}
				}
				break;
		    }
			case 6:{
				for (j=0;j<in;j++){
					for (k=0;k<m;k++){
						dark = 1.;
						for (i=0;i<dDim0;i++)
							dark *= pow(w[i*m+k],d[i]);
						F[m*j+k] = dark2*dark*( 6.*( sig[m*j+k]-31.*pow(sig[m*j+k],2)+180.*pow(sig[m*j+k],3)-390.*pow(sig[m*j+k],4)+360.*pow(sig[m*j+k],5)-120.*pow(sig[m*j+k],6) ) + zint[m*j+k] * ( sig[m*j+k]-63.*pow(sig[m*j+k],2)+602.*pow(sig[m*j+k],3)-2100.*pow(sig[m*j+k],4)+3360.*pow(sig[m*j+k],5)-2520.*pow(sig[m*j+k],6)+720.*pow(sig[m*j+k],7) ));
					}
				}
				break;
		    }
			case 7:{
				for (j=0;j<in;j++){
					for (k=0;k<m;k++){
						dark = 1.;
						for (i=0;i<dDim0;i++)
							dark *= pow(w[i*m+k],d[i]);
						F[m*j+k] = dark2*dark*( 7.*( sig[m*j+k]-63.*pow(sig[m*j+k],2)+602.*pow(sig[m*j+k],3)-2100.*pow(sig[m*j+k],4)+3360.*pow(sig[m*j+k],5)-2520.*pow(sig[m*j+k],6)+720.*pow(sig[m*j+k],7) ) + zint[m*j+k] * ( sig[m*j+k]-127.*pow(sig[m*j+k],2)+1932.*pow(sig[m*j+k],3)-10206.*pow(sig[m*j+k],4)+25200.*pow(sig[m*j+k],5)-31920.*pow(sig[m*j+k],6)+20160.*pow(sig[m*j+k],7)-5040.*pow(sig[m*j+k],8) ));
					}
				}
				break;
		    }
			case 8:{
				for (j=0;j<in;j++){
					for (k=0;k<m;k++){
						dark = 1.;
						for (i=0;i<dDim0;i++)
							dark *= pow(w[i*m+k],d[i]);
						F[m*j+k] = dark2*dark*( 8.*( sig[m*j+k]-127.*pow(sig[m*j+k],2)+1932.*pow(sig[m*j+k],3)-10206.*pow(sig[m*j+k],4)+25200.*pow(sig[m*j+k],5)-31920.*pow(sig[m*j+k],6)+20160.*pow(sig[m*j+k],7)-5040.*pow(sig[m*j+k],8) ) + zint[m*j+k] * ( sig[m*j+k]-255.*pow(sig[m*j+k],2)+6050.*pow(sig[m*j+k],3)-46620.*pow(sig[m*j+k],4)+166824.*pow(sig[m*j+k],5)-317520.*pow(sig[m*j+k],6)+332640.*pow(sig[m*j+k],7)-181440.*pow(sig[m*j+k],8)+40320.*pow(sig[m*j+k],9) ));
					}
				}
				break;
		    }
		}
	}
	delete[] sig; delete[] zint; 
	return;
};

// nELM Swish *******************************************************************************************
void nELMReLU::nElmHint(const int* d, int dDim0, const double* x, const int in, double* F){

	int i,j,k,h=-1;
	double dark,dark1;
	bool zeroFlag=false;
	for (i=0;i<dDim0;i++){
		if (d[i] > 1 || ((d[i] == 1) && (h != -1))){
			zeroFlag = true;
			break;
		} else if (d[i] == 1) {
			h = i;
			dark1 = c[i];
		}
	}
	
	if (zeroFlag) {
		for (j=0;j<in;j++){
			for (k=0;k<m;k++)
				F[m*j+k] = 0.;
		}
	} else {
		for (j=0;j<in;j++){
			for (k=0;k<m;k++){
				dark = 0.;
				for (i=0;i<dim;i++)
					dark += w[i*m+k]*x[i*in+j];
				F[m*j+k] = std::max(0.,dark+b[k]);
			}
		}
	} 

	if (h != -1){
		for (j=0;j<in;j++){
			for (k=0;k<m;k++){
				if (F[m*j+k] != 0.){
					F[m*j+k] = dark1*w[h*m+k];
				}
			}
		}
	}
	return;
};
