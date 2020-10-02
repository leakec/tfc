#include <iostream>
#include <math.h>
#include <Python.h>

#ifndef BF_H
#define BF_H

// CP **************************************************************************************************************************
class CP {
	public:
		static double* z;
		static double c;
		static int* nC;
		static int numC, n, m;

	public:
		CP(double* zin, int zDim0, int* nCin, int ncDim0, int min, double cin);
		~CP();

		/** Function is used to create a Chebyshev Polynomail (CP) matrix and its derivatives. This matrix is is an m x N matrix where
		 *  m is the number of basis functions and N is the number of points in x. The argument d is used to specify the derivative.
		 *  Note that this function is used to hook into Python, thus the extra arguments. */
		void H(double* x, int in, const int d, int* nOut, int* mOut, double** F,  bool full, bool useVal);

		/** This function creates a PyCapsule object that wraps the XLA verison of the basis function. */
		PyObject* GetXlaCapsule();

	private:
		/** Function used internally to create the univariate CP matrices. */
		static void Hint(const int d, const double* x, const int nOut, const int mOut, double*& F, bool full);

		/** Function used internally to create derivatives of CP matrices. */
		static void RecurseDeriv(const int d, int dCurr, const double* x, const int nOut, double*& F, const int mOut);

		/** This function is an XLA version of CP. */
		static void xla(void* out, void** in);
		
};

#endif
