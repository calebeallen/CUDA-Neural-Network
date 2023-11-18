
#include <iostream>
#include <ctime>
#include "Linear.cuh"
#include "curand_kernel.h"

using namespace std;

__global__ void d_set(double* A, int i, double val) {

    A[i] = val;

}

__global__ void d_add( double* A, double* B, double* C, int xD, int yD, int size ) {

    // z * ydim * xdim + y * xdim + x
    int i = ((blockIdx.z * blockDim.z + threadIdx.z) * yD + (blockIdx.y * blockDim.y + threadIdx.y)) * xD + (blockIdx.x * blockDim.x + threadIdx.x);

    if ( i < size )

        C[i] = A[i] + B[i];

}

__global__ void d_sub(double* A, double* B, double* C, int xD, int yD, int size) {

    // z * ydim * xdim + y * xdim + x
    int i = ((blockIdx.z * blockDim.z + threadIdx.z) * yD + (blockIdx.y * blockDim.y + threadIdx.y)) * xD + (blockIdx.x * blockDim.x + threadIdx.x);

    if (i < size)

        C[i] = A[i] - B[i];

}

__global__ void d_addScalar(double* A, double scalar, int xD, int yD, int size) {

    int i = ((blockIdx.z * blockDim.z + threadIdx.z) * yD + (blockIdx.y * blockDim.y + threadIdx.y)) * xD + (blockIdx.x * blockDim.x + threadIdx.x);

    if (i < size)

        A[i] += scalar;

}


__global__ void d_multiplyScalar(double* A, double scalar, int xD, int yD, int size) {

    int i = ((blockIdx.z * blockDim.z + threadIdx.z) * yD + (blockIdx.y * blockDim.y + threadIdx.y)) * xD + (blockIdx.x * blockDim.x + threadIdx.x);

    if (i < size)

        A[i] *= scalar;

}


__global__ void d_fill( double* A, double val, int xD, int yD, int size ) {

    int i = ((blockIdx.z * blockDim.z + threadIdx.z) * yD + (blockIdx.y * blockDim.y + threadIdx.y)) * xD + (blockIdx.x * blockDim.x + threadIdx.x);

    if ( i < size )

        A[i] = val;

}

__global__ void d_multiplyAdd(double* A, double* B, double* C, int xD, int yD, int l) {

    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < xD && y < yD)

        for (int i = 0; i < l; i++)

            C[y * xD + x] += A[i * xD + x] * B[y * l + i];

}

__global__ void d_multiplySet(double* A, double* B, double* C, int xD, int yD, int l) {

    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;

    double dotProduct = 0;

    if (x < xD && y < yD) {

        for (int i = 0; i < l; i++)

            dotProduct += A[i * xD + x] * B[y * l + i];

        C[y * xD + x] = dotProduct;

    }

}

Linear::Linear( int _xD, int _yD, int _size ) {

    xD = _xD;

    yD = _yD;

    size = _size;

    cudaMalloc( (void**)&d_self, size * sizeof(double) );

    cudaMemset( d_self, 0, size * sizeof(double) );

}


Linear::~Linear() {

    cudaFree( d_self );

}

void Linear::idxSet(int idx, double val) {

    d_set <<< 1, 1 >>> (d_self, idx, val);

}

void Linear::add( Linear* A, Linear* B ) {

    if(B->size != A->size || A->size != size)

        throw invalid_argument("Invalid size.");

    d_add <<< gridDim, blockDim >>> ( A->d_self, B->d_self, d_self, xD, yD, size );

}

void Linear::add(Linear* B){

    add(this, B);

}

void Linear::sub(Linear* A, Linear* B) {

    if (B->size != A->size || A->size != size)

        throw invalid_argument("Invalid size.");

    d_sub <<< gridDim, blockDim >>> (A->d_self, B->d_self, d_self, xD, yD, size);

}

void Linear::sub(Linear* B) {

    sub(this, B);

}

void Linear::multiplyAdd( Linear* A, Linear* B ) {

    if (A->yD != B->xD || xD != A->xD || yD != B->yD)

        throw invalid_argument("Invalid multiplication operation.");

    d_multiplyAdd <<< gridDim, blockDim >>> ( A->d_self, B->d_self, d_self, xD, yD, A->yD );

}


void Linear::multiplySet(Linear* A, Linear* B) {

    if (A->yD != B->xD || xD != A->xD || yD != B->yD)

        throw invalid_argument("Invalid multiplication operation.");

    d_multiplySet <<< gridDim, blockDim >>> (A->d_self, B->d_self, d_self, xD, yD, A->yD );

}


void Linear::addScalar(double scalar) {

    d_addScalar <<< gridDim, blockDim >>> (d_self, scalar, xD, yD, size);

}


void Linear::multiplyScalar(double scalar) {

    d_multiplyScalar <<< gridDim, blockDim >>> (d_self, scalar, xD, yD, size);

}


void Linear::fill(double val) {

    d_fill <<< gridDim, blockDim >>> ( d_self, val, xD, yD, size );

}


void Linear::copy(double* A) {

    cudaMemcpy(d_self, A, size * sizeof(double), cudaMemcpyHostToDevice);

}

void Linear::copyTo(double* A) {

    cudaMemcpy(A, d_self, size * sizeof(double), cudaMemcpyDeviceToHost);

}

void Linear::randomize(double min, double max) {

    srand(time(NULL));

    double* arr = new double[size];

    for(int i = 0; i < size; i++)

        arr[i] = (double)(rand()) / (RAND_MAX + 1) * (max - min) + min;

    cudaMemcpy(d_self, arr, size * sizeof(double), cudaMemcpyHostToDevice);

}


Matrix::Matrix(int _cols, int _rows) : Linear(_cols, _rows, _cols* _rows) {

    cols = _cols;

    rows = _rows;

    gridDim = dim3(_cols / 32 + 1, rows / 32 + 1, 1);

    blockDim = dim3(32, 32, 1);

}


void Matrix::set(int x, int y, double val) {

    idxSet(y * xD + x, val);

}


void Matrix::print() {

    double* arr = new double[rows * cols];

    cudaMemcpy(arr, d_self, rows * cols * sizeof(double), cudaMemcpyDeviceToHost);

    cout << "\n";

    for (int r = 0; r < rows; r++) {

        for (int c = 0; c < cols; c++)

            cout << arr[r * cols + c] << " ";

        cout << "\n";

    }

    delete[] arr;

}


Vector::Vector(int _length, bool _isCol) : Linear(0, 0, _length) {

    length = _length;
    
    if (_isCol)

        column();

    else

        row();

}


Vector::Vector(int _length) : Vector(_length, true) {}


void Vector::set(int i, double val) {

    idxSet(i, val);

}


void Vector::row() {

    isCol = false;

    gridDim = dim3(length / 1024 + 1, 1, 1);

    blockDim = dim3(1024, 1, 1);

    xD = length;

    yD = 1;

}


void Vector::column() {

    isCol = true;

    gridDim = dim3(1, length / 1024 + 1, 1);

    blockDim = dim3(1, 1024, 1);

    xD = 1;

    yD = length;

}


void Vector::print() {

    double* arr = new double[length];

    cudaMemcpy(arr, d_self, length * sizeof(double), cudaMemcpyDeviceToHost);

    cout << "\n";

    for (int i = 0; i < length; i++)

        cout << arr[i] << (isCol ? "\n" : " ");

    if (!isCol)

        cout << "\n";

    delete[] arr;

}
