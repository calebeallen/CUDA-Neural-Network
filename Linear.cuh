
#ifndef LINEAR
#define LINEAR

#include "cuda_runtime.h"

class Linear {

	protected:

		void idxSet(int, double);

	public:

		int size, xD, yD;

		dim3 blockDim, gridDim;

		double* d_self;

		Linear(int,int,int);

		~Linear();

		void add( Linear* );

		void add( Linear*, Linear* );

		void sub( Linear* );

		void sub( Linear*, Linear* );

		void multiplyAdd( Linear*, Linear* );

		void multiplySet( Linear*, Linear* );

		void addScalar(double);

		void multiplyScalar(double);

		void fill(double);

		void copy(double*);

		void copyTo(double*);

		void randomize(double, double);

};


class Vector : public Linear {

	public:

		bool isCol;

		int length;

		Vector(int, bool);

		Vector(int);

		void set(int, double);

		void row();

		void column();

		void print();

};


class Matrix : public Linear {

	public:

		int rows, cols;

		Matrix(int, int);

		void set(int, int, double);

		void print();

};


#endif