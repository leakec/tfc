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


#ifndef BF_H
#define BF_H

// BasisFunc **************************************************************************************************************************
/** This class is an abstract class used to create all other basis function classes. It defines standard methods to call the basis function and its
 *  derivatives, as well as provides wrappers for XLA computation. */
class BasisFunc{

	public:
		/** Beginning of the basis function domain. */
		double z0;

		/** Start of the problem domain. */
		double x0;

		/** Multiplier in the linear domain map. */
		double c;

		/** Array that specifies which basis functions to remove. */
		int* nC;

		/** Number of basis functions to be removed. */
		int numC;

		/** Number of basis functions to use. */
		int m;

		/** Unique identifier for this instance of BasisFunc. */
		int identifier;

		/** PyObject that contains the XLA version of the basis function. */
		PyObject* xlaCapsule;

		#ifdef HAS_CUDA
			/** PyObject that contains the XLA version of the basis function that uses a CUDA GPU kernel. */
			PyObject* xlaGpuCapsule;
		#else
			const char* xlaGpuCapsule = "CUDA NOT FOUND, GPU NOT IMPLEMENTED.";
		#endif

		/** Counter that increments each time a new instance of BasisFunc is created. */
		static int nIdentifier;

		/** Vector that contains pointers to all BasisFunc classes. */
		static std::vector<BasisFunc*> BasisFuncContainer;

	public:
		/** Basis function class constructor. 
		 * 		- Stores variables based on user supplied givens
		 * 		- Stores a pointer to itself using static variables
		 * 		- Creates PyCapsule for xla function. */
		BasisFunc(double x0in, double xf, int* nCin, int ncDim0, int min, double z0in=0., double zf=DBL_MAX);

		/** Dummy empty constructor allows derived classes without calling constructor explicitly. */
		BasisFunc(){};

		/** Basis function class destructor. Removes memory used by the basis function class. */
		virtual ~BasisFunc();

		/** Function is used to create a basis function matrix and its derivatives. This matrix is is an m x N matrix where:
		 *  	- m is the number of basis functions
		 *  	- N = in is the number of points in x
		 *  	- d is used to specify the derivative
		 *  	- full is a bool that specifies:
		 *  		- If true, full matrix with no basis functions removed is returned
		 *  		- If false, matrix columns corresponding to the values in nC are removed
		 *  	- useVal is a bool that specifies:
		 *  		- If true, uses the x values given
		 *  		- If false, uses the z values from the class
		 *  Note that this function is used to hook into Python, thus the extra arguments. */
		virtual void H(double* x, int n, const int d, int* nOut, int* mOut, double** F,  bool full);

		/** This function is an XLA version of the basis function. */
		virtual void xla(void* out, void** in);

		#ifdef HAS_CUDA
			/** This function is an XLA version of the basis function that uses a CUDA GPU kernel. */
			void xlaGpu(CUstream stream, void** buffers, const char* opaque, size_t opaque_len);
		#endif
	
	protected:
		/** This function creates a PyCapsule object that wraps the XLA verison of the basis function. */
		PyObject* GetXlaCapsule();

		#ifdef HAS_CUDA
			/** This function creates a PyCapsule object that wraps the XLA verison of the basis function that uses a CUDA GPU kernel. */
			PyObject* GetXlaCapsuleGpu();
		#endif

	private:
		/** Function used internally to create the basis function matrices. */
		virtual void Hint(const int d, const double* x, const int nOut, double* dark) = 0;

		/** Function used internally to create derivatives of the basis function matrices. */
		virtual void RecurseDeriv(const int d, int dCurr, const double* x, const int nOut, double*& F, const int mOut) = 0;
};

// XLA related declarations: **********************************************************************************************************
/** Pointer for XLA-type function that can be cast to void* and put in a PyCapsule. */
typedef void(*xlaFnType)(void*,void**);

#ifdef HAS_CUDA
	/** Pointer for GPU compatible XLA-type function that can be cast to void* and put in a PyCapsule. */
	typedef void(*xlaGpuFnType)(CUstream,void**,const char*,size_t);
	
	/** Function used to wrap BasisFunc->xlaGpu in C-style function that can be cast to void*. */
	void xlaGpuWrapper(CUstream stream, void** buffers, const char* opaque, size_t opaque_len);
#endif

// CP: ********************************************************************************************************************************
/** Class for Chebyshev orthogonal polynomials. */
class CP: virtual public BasisFunc {
	public:
		/** CP class constructor. Calls BasisFunc class constructor. See BasisFunc class for more details. */
		CP(double x0, double xf, int* nCin, int ncDim0, int min):
		  BasisFunc(x0,xf,nCin,ncDim0,min,-1.,1.){};

		/** Dummy CP class constructor. Used only in n-dimensions. */
		CP(){};

		/** CP class destructor.*/
		virtual ~CP(){};

	protected:
		/** Function used internally to create the basis function matrices. */
		void Hint(const int d, const double* x, const int nOut, double* dark);

		/** Function used internally to create derivatives of the basis function matrices. */
		void RecurseDeriv(const int d, int dCurr, const double* x, const int nOut, double*& F, const int mOut);
};

// LeP: ********************************************************************************************************************************
/** Class for Legendre orthogonal polynomials. */
class LeP: virtual public BasisFunc {
	public:
		/** LeP class constructor. Calls BasisFunc class constructor. See BasisFunc class for more details. */
		LeP(double x0, double xf, int* nCin, int ncDim0, int min):
		  BasisFunc(x0,xf,nCin,ncDim0,min,-1.,1.){};

		/** Dummy LeP class constructor. Used only in n-dimensions. */
		LeP(){};

		/** LeP class destructor.*/
		~LeP(){};

	protected:
		/** Function used internally to create the basis function matrices. */
		void Hint(const int d, const double* x, const int nOut, double* dark);

		/** Function used internally to create derivatives of the basis function matrices. */
		void RecurseDeriv(const int d, int dCurr, const double* x, const int nOut, double*& F, const int mOut);
};

// LaP: ********************************************************************************************************************************
/** Class for Laguerre orthogonal polynomials. */
class LaP: public BasisFunc {
	public:
		/** LaP class constructor. Calls BasisFunc class constructor. See BasisFunc class for more details. */
		LaP(double x0, double xf, int* nCin, int ncDim0, int min):
		  BasisFunc(x0,xf,nCin,ncDim0,min){};
		/** LaP class destructor.*/
		~LaP(){};

	private:
		/** Function used internally to create the basis function matrices. */
		void Hint(const int d, const double* x, const int nOut, double* dark);

		/** Function used internally to create derivatives of the basis function matrices. */
		void RecurseDeriv(const int d, int dCurr, const double* x, const int nOut, double*& F, const int mOut);
};

// HoPpro: ********************************************************************************************************************************
/** Class for Hermite probablist orthogonal polynomials. */
class HoPpro: public BasisFunc {
	public:
		/** HoPpro class constructor. Calls BasisFunc class constructor. See BasisFunc class for more details. */
		HoPpro(double x0, double xf, int* nCin, int ncDim0, int min):
		  BasisFunc(x0,xf,nCin,ncDim0,min){};
		/** HoPpro class destructor.*/
		~HoPpro(){};

	private:
		/** Function used internally to create the basis function matrices. */
		void Hint(const int d, const double* x, const int nOut, double* dark);

		/** Function used internally to create derivatives of the basis function matrices. */
		void RecurseDeriv(const int d, int dCurr, const double* x, const int nOut, double*& F, const int mOut);
};

// HoPphy: ********************************************************************************************************************************
/** Class for Hermite physicist orthogonal polynomials. */
class HoPphy: public BasisFunc {
	public:
		/** HoPphy class constructor. Calls BasisFunc class constructor. See BasisFunc class for more details. */
		HoPphy(double x0, double xf, int* nCin, int ncDim0, int min):
		  BasisFunc(x0,xf,nCin,ncDim0,min){};
		/** HoPphy class destructor.*/
		~HoPphy(){};

	private:
		/** Function used internally to create the basis function matrices. */
		void Hint(const int d, const double* x, const int nOut, double* dark);

		/** Function used internally to create derivatives of the basis function matrices. */
		void RecurseDeriv(const int d, int dCurr, const double* x, const int nOut, double*& F, const int mOut);
};

// FS: ********************************************************************************************************************************
/** Class for Fourier Series basis. */
class FS: virtual public BasisFunc {
	public:
		/** FS class constructor. Calls BasisFunc class constructor. See BasisFunc class for more details. */
		FS(double x0, double xf, int* nCin, int ncDim0, int min):
		  BasisFunc(x0,xf,nCin,ncDim0,min,-M_PI,M_PI){};

		/** Dummy FS class constructor. Used only in n-dimensions. */
		FS(){};

		/** FS class destructor.*/
		~FS(){};

	protected:
		/** Function used internally to create the basis function matrices and derivatives. */
		void Hint(const int d, const double* x, const int nOut, double* dark);

		/** This function is unecessary for FS as it is all handled in Hint. Therefore, this is just an empty function that returns a warning. */
		void RecurseDeriv(const int d, int dCurr, const double* x, const int nOut, double*& F, const int mOut){
			fprintf(stderr, "Warning, this function from FS should never be called. It seems it has been called by accident. Please check that this function was intended to be called.\n");
			printf("Warning, this function from FS should never be called. It seems it has been called by accident. Please check that this function was intended to be called.\n");
		};

};

// ELM base class: ********************************************************************************************************************************
/** ELM base class. */
class ELM: public BasisFunc {
	public:
		/** ELM weights. */
		double *w;

		/** ELM biases. */
		double *b; 

		/** ELM class constructor. Calls BasisFunc class constructor and sets up weights and biases for the ELM. See BasisFunc class for more details. */
		ELM(double x0, double xf, int* nCin, int ncDim0, int min);

		/** ELM class destructor.*/
		virtual ~ELM();

		/** Python hook to return ELM weights. */
		void getW(double** arrOut, int* nOut);

		/** Python hook to set ELM weights. */
		void setW(double* arrIn, int nIn);

		/** Python hook to return ELM biases. */
		void getB(double** arrOut, int* nOut);

		/** Python hook to set ELM biases. */
		void setB(double* arrIn, int nIn);

	protected:
		/** Function used internally to create the basis function matrices. */
		virtual void Hint(const int d, const double* x, const int nOut, double* dark) = 0;

		/** This function is unecessary for ELM as it is all handled in Hint. Therefore, this is just an empty function that returns a warning. */
		void RecurseDeriv(const int d, int dCurr, const double* x, const int nOut, double*& F, const int mOut){
			fprintf(stderr, "Warning, this function from ELM should never be called. It seems it has been called by accident. Please check that this function was intended to be called.\n");
			printf("Warning, this function from ELM should never be called. It seems it has been called by accident. Please check that this function was intended to be called.\n");
		};

};

// ELM sigmoid: ********************************************************************************************************************************
/** ELM that uses the sigmoid activation function. */
class ELMSigmoid: public ELM {

	public:
		/** ELMSigmoid class constructor. Calls ELM class constructor. See ELM class for more details. */
		ELMSigmoid(double x0, double xf, int* nCin, int ncDim0, int min):
		  ELM(x0,xf,nCin,ncDim0,min){};

		/** ELMSigmoid class destructor.*/
		~ELMSigmoid(){};

	protected:
		/** Function used internally to create the basis function matrices and derivatives. */
		void Hint(const int d, const double* x, const int nOut, double* dark);

};

// ELM ReLU: ********************************************************************************************************************************
/** ELM that uses the recitified linear unit activation function. */
class ELMReLU: public ELM {

	public:
		/** ELMReLU class constructor. Calls ELM class constructor. See ELM class for more details. */
		ELMReLU(double x0, double xf, int* nCin, int ncDim0, int min):
		  ELM(x0,xf,nCin,ncDim0,min){};

		/** ELMReLU class destructor.*/
		~ELMReLU(){};

	protected:
		/** Function used internally to create the basis function matrices and derivatives. */
		void Hint(const int d, const double* x, const int nOut, double* dark);

};

// ELM Tanh: ********************************************************************************************************************************
/** ELM that uses the tanh activation function. */
class ELMTanh: public ELM {

	public:
		/** ELMTanh class constructor. Calls ELM class constructor. See ELM class for more details. */
		ELMTanh(double x0, double xf, int* nCin, int ncDim0, int min):
		  ELM(x0,xf,nCin,ncDim0,min){};

		/** ELMTanh class destructor.*/
		~ELMTanh(){};

	private:
		/** Function used internally to create the basis function matrices and derivatives. */
		void Hint(const int d, const double* x, const int nOut, double* dark);

};

// ELM Sin: ********************************************************************************************************************************
/** ELM that uses the sin activation function. */
class ELMSin: public ELM {

	public:
		/** ELMSin class constructor. Calls ELM class constructor. See ELM class for more details. */
		ELMSin(double x0, double xf, int* nCin, int ncDim0, int min):
		  ELM(x0,xf,nCin,ncDim0,min){};

		/** ELMSin class destructor.*/
		~ELMSin(){};

	private:
		/** Function used internally to create the basis function matrices and derivatives. */
		void Hint(const int d, const double* x, const int nOut, double* dark);

};

// ELM Swish: ********************************************************************************************************************************
/** ELM that uses the swish activation function. */
class ELMSwish: public ELM {

	public:
		/** ELMSwish class constructor. Calls ELM class constructor. See ELM class for more details. */
		ELMSwish(double x0, double xf, int* nCin, int ncDim0, int min):
		  ELM(x0,xf,nCin,ncDim0,min){};

		/** ELMSwish class destructor.*/
		~ELMSwish(){};

	private:
		/** Function used internally to create the basis function matrices and derivatives. */
		void Hint(const int d, const double* x, const int nOut, double* dark);

};

// n-D Basis function base class: ***************************************************************************************************

/** Base class for n-dimensional basis functions. This class inherits from BasisFunc, and contains
 *  methods that are used for all n-dimensional basis fuctions. This is an abstract class. 
 *  Concrete n-dimensional basis functions will inherit from this class and one of the concrete
 *  1-dimensional basis function classes. */
class nBasisFunc: virtual public BasisFunc{

	public:

		/** Beginning of the basis function domain. */
		double z0;

		/** Beginning of the basis function domain. */
		double zf; 

		/** Multipliers for the linear domain map. */
		double* c;

		/** Initial value of the domain */
		double* x0;

		/** Number of dimensions. */
		int dim;

		/** Number of basis functions in H matrix. */
		int numBasisFunc;

		/** Number of basis functions in full H matrix. */
		int numBasisFuncFull;

	public:
		/** n-D basis function class constructor. */
		nBasisFunc(double* x0in, int x0Dim0, double* xf, int xfDim0, int* nCin, int ncDim0, int ncDim1, int min, double z0in=0., double zfin=0.);

		/** Dummy nBasisFunc constructor used by nELM only. */
		nBasisFunc(){};

		/** n-D basis function class destructor. */
		virtual ~nBasisFunc();

		/** This function is used to create a basis function matrix and its derivatives. */
		void H(double* x, int in, int xDim1, int* d, int dDim0, int* nOut, int* mOut, double** F, const bool full);

		/** This function is an XLA version of the basis function. */
		void xla(void* out, void** in);

		/** Python hook to return domain mapping constants. */
		void getC(double** arrOut, int* nOut);

	private:
		/** Recursive function used to perform the tensor product of univarite basis functions to form multivariate basis functions. */
		void RecurseBasis(int dimCurr, int* vec, int &count, const bool full, const int in, const int numBasis, const double* T, double* out);

		/** Recursive function used to calculate the size of the multivariate basis function matrix. */
		void NumBasisFunc(int dimCurr, int* vec, int &count, const bool full);

		/** Internal function used to calculate dim sets of univariate basis functions with specified derivatives. Note, that if dDim0 < dim, then 0's will be used for the tail end.*/
		virtual void nHint(double* x, int in, const int* d, int dDim0, int numBasis, double*& F, const bool full);

		/** Function used internally to create the basis function matrices. */
		virtual void Hint(const int d, const double* x, const int nOut, double* dark) = 0;

		/** Function used internally to create derivatives of the basis function matrices. */
		virtual void RecurseDeriv(const int d, int dCurr, const double* x, const int nOut, double*& F, const int mOut) = 0;

};

// n-D CP class: ******************************************************************************************************************
/** Class for n-dimensional Chebyshev orthogonal polynomials. */
class nCP: public nBasisFunc, public CP {
	
	public:

		/** nCP class constructor. Calls nBasisFunc class constructor and dummy constructors of remaining parents. See nBasisFunc class for more details. */
		nCP(double* x0in, int x0Dim0, double* xf, int xfDim0, int* nCin, int ncDim0, int ncDim1, int min):nBasisFunc(x0in,x0Dim0,xf,xfDim0,nCin,ncDim0,ncDim1,min,-1.,1.){};

		/** nCP class destructor.*/
		~nCP(){};

	private:
		/** Function used internally to create the basis function matrices. */
		void Hint(const int d, const double* x, const int nOut, double* dark){CP::Hint(d,x,nOut,dark);};

		/** Function used internally to create derivatives of the basis function matrices. */
		void RecurseDeriv(const int d, int dCurr, const double* x, const int nOut, double*& F, const int mOut){CP::RecurseDeriv(d,dCurr,x,nOut,F,mOut);};


};

// n-D LeP class: ******************************************************************************************************************
/** Class for n-dimensional Legendre orthogonal polynomials. */
class nLeP: public nBasisFunc, public LeP {
	
	public:
		/** nLeP class constructor. Calls nBasisFunc class constructor and dummy constructors of remaining parents. See nBasisFunc class for more details. */
		nLeP(double* x0in, int x0Dim0, double* xf, int xfDim0, int* nCin, int ncDim0, int ncDim1, int min):nBasisFunc(x0in,x0Dim0,xf,xfDim0,nCin,ncDim0,ncDim1,min,-1.,1.){};

		/** nLeP class destructor.*/
		~nLeP(){};

	private:
		/** Function used internally to create the basis function matrices. */
		void Hint(const int d, const double* x, const int nOut, double* dark){LeP::Hint(d,x,nOut,dark);};

		/** Function used internally to create derivatives of the basis function matrices. */
		void RecurseDeriv(const int d, int dCurr, const double* x, const int nOut, double*& F, const int mOut){LeP::RecurseDeriv(d,dCurr,x,nOut,F,mOut);};

};

// n-D FS class: ******************************************************************************************************************
/** Class for n-dimensional Fourier Series. */
class nFS: public nBasisFunc, public FS {
	
	public:
		/** nFS class constructor. Calls nBasisFunc class constructor and dummy constructors of remaining parents. See nBasisFunc class for more details. */
		nFS(double* x0in, int x0Dim0, double* xf, int xfDim0, int* nCin, int ncDim0, int ncDim1, int min):nBasisFunc(x0in,x0Dim0,xf,xfDim0,nCin,ncDim0,ncDim1,min,-M_PI,M_PI){};

		/** nFS class destructor.*/
		~nFS(){};

	private:
		/** Function used internally to create the basis function matrices. */
		void Hint(const int d, const double* x, const int nOut, double* dark){FS::Hint(d,x,nOut,dark);};

		/** Function used internally to create derivatives of the basis function matrices. */
		void RecurseDeriv(const int d, int dCurr, const double* x, const int nOut, double*& F, const int mOut){FS::RecurseDeriv(d,dCurr,x,nOut,F,mOut);};

};

// n-D ELM base class: *******************************************************************************************************************************************************
/** n-D ELM base class. */
class nELM: public nBasisFunc {

	public:
		/** Beginning of the basis function domain. */
		double z0;

		/** Beginning of the basis function domain. */
		double zf; 

		/** nELM weights. */
		double *w;

		/** nELM biases. */
		double *b; 

		/** n-D ELM class constructor. */
		nELM(double* x0in, int x0Dim0, double* xf, int xfDim0, int* nCin, int ncDim0, int min, double z0in=0., double zfin=1.);

		/** n-D ELM class destructor. */
		virtual ~nELM();

		/** Python hook to return nELM weights. */
		void setW(double* arrIn, int dimIn, int nIn);

		/** Python hook to set nELM weights. */
		void getW(int* dimOut, int* nOut, double** arrOut);

		/** Python hook to return nELM biases. */
		void getB(double** arrOut, int* nOut);

		/** Python hook to set nELM biases. */
		void setB(double* arrIn, int nIn);

	private:

		/** Internal function used to calculate dim sets of univariate basis functions with specified derivatives. Note, that if dDim0 < dim, then 0's will be used for the tail end.*/
		void nHint(double* x, int in, const int* d, int dDim0, int numBasis, double*& F, const bool full) override;

		/** This function handles creating a full matrix of nELM basis functions. */
		virtual void nElmHint(const int* d, int dDim0, const double* x, const int in, double* F) = 0;

		/** This function is unecessary for nELM as it is all handled in nElmHint. Therefore, this is just an empty function that returns a warning. */
		void Hint(const int d, const double* x, const int nOut, double* dark) override {
			fprintf(stderr, "Warning, this function from nELM should never be called. It seems it has been called by accident. Please check that this function was intended to be called.\n");
			printf("Warning, this function from nELM should never be called. It seems it has been called by accident. Please check that this function was intended to be called.\n");
		};

		/** This function is unecessary for nELM as it is all handled in nElmHint. Therefore, this is just an empty function that returns a warning. */
		void RecurseDeriv(const int d, int dCurr, const double* x, const int nOut, double*& F, const int mOut) override {
			fprintf(stderr, "Warning, this function from nELM should never be called. It seems it has been called by accident. Please check that this function was intended to be called.\n");
			printf("Warning, this function from nELM should never be called. It seems it has been called by accident. Please check that this function was intended to be called.\n");
		};

};

// n-D ELM sigmoid class: *******************************************************************************************************************************************************
/** n-D ELM that uses the sigmoid activation function. */
class nELMSigmoid: public nELM {
	
	public:
		/** nELMSigmoid class constructor. Calls nELM class constructor. See nELM class for more details. */
		nELMSigmoid(double* x0in, int x0Dim0, double* xf, int xfDim0, int* nCin, int ncDim0,int min):nELM(x0in,x0Dim0,xf,xfDim0,nCin,ncDim0,min){};

		/** nELMSigmoid class destructor.*/
		~nELMSigmoid(){};

	private:
		/** Function used internally to create the basis function matrices. */
		void nElmHint(const int* d, int dDim0, const double* x, const int in, double* F) override ;
};

// n-D ELM Tanh class: *******************************************************************************************************************************************************
/** n-D ELM that uses the tanh activation function. */
class nELMTanh: public nELM {
	
	public:
		/** nELMTanh class constructor. Calls nELM class constructor. See nELM class for more details. */
		nELMTanh(double* x0in, int x0Dim0, double* xf, int xfDim0, int* nCin, int ncDim0, int min):nELM(x0in,x0Dim0,xf,xfDim0,nCin,ncDim0,min){};

		/** nELMTanh class destructor.*/
		~nELMTanh(){};

	private:
		/** Function used internally to create the basis function matrices. */
		void nElmHint(const int* d, int dDim0, const double* x, const int in, double* F) override ;
};

// n-D ELM Sin class: *******************************************************************************************************************************************************
/** n-D ELM that uses the sine activation function. */
class nELMSin: public nELM {
	
	public:
		/** nELMSin class constructor. Calls nELM class constructor. See nELM class for more details. */
		nELMSin(double* x0in, int x0Dim0, double* xf, int xfDim0, int* nCin, int ncDim0, int min):nELM(x0in,x0Dim0,xf,xfDim0,nCin,ncDim0,min){};

		/** nELMSin class destructor.*/
		~nELMSin(){};

	private:
		/** Function used internally to create the basis function matrices. */
		void nElmHint(const int* d, int dDim0, const double* x, const int in, double* F) override ;
};

// n-D ELM Swish class: *******************************************************************************************************************************************************
/** n-D ELM that uses the swish activation function. */
class nELMSwish: public nELM {
	
	public:
		/** nELMSwish class constructor. Calls nELM class constructor. See nELM class for more details. */
		nELMSwish(double* x0in, int x0Dim0, double* xf, int xfDim0, int* nCin, int ncDim0, int min):nELM(x0in,x0Dim0,xf,xfDim0,nCin,ncDim0,min){};

		/** nELMSwish class destructor.*/
		~nELMSwish(){};

	private:
		/** Function used internally to create the basis function matrices. */
		void nElmHint(const int* d, int dDim0, const double* x, const int in, double* F) override ;
};

// n-D ELM ReLU class: *******************************************************************************************************************************************************
/** n-D ELM that uses the rectified linear activation function. */
class nELMReLU: public nELM {
	
	public:
		/** nELMReLU class constructor. Calls nELM class constructor. See nELM class for more details. */
		nELMReLU(double* x0in, int x0Dim0, double* xf, int xfDim0, int* nCin, int ncDim0, int min):nELM(x0in,x0Dim0,xf,xfDim0,nCin,ncDim0,min){};

		/** nELMReLU class destructor.*/
		~nELMReLU(){};

	private:
		/** Function used internally to create the basis function matrices. */
		void nElmHint(const int* d, int dDim0, const double* x, const int in, double* F) override ;
};
	
#endif
