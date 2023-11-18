
#include "NeuralNetwork.cuh"
#include <iostream>
#include <iomanip>

using namespace std;

__global__ void d_lReLU(double* A, double a, int xD, int yD, int size) {

    int i = ((blockIdx.z * blockDim.z + threadIdx.z) * yD + (blockIdx.y * blockDim.y + threadIdx.y)) * xD + (blockIdx.x * blockDim.x + threadIdx.x);

    if (i < size) {

        double a1 = A[i], a2 = a1 * a;

        A[i] = a1 > a2 ? a1 : a2;

    }

}

__global__ void d_lReLU_Deriv(double* A, double* B, double a, int xD, int yD, int size) {

    int i = ((blockIdx.z * blockDim.z + threadIdx.z) * yD + (blockIdx.y * blockDim.y + threadIdx.y)) * xD + (blockIdx.x * blockDim.x + threadIdx.x);

    if (i < size)

        B[i] *= A[i] <= 0 ? a : 1;

}

NeuralNetwork::NeuralNetwork(int _layers, int* nodeCount, double _learnRate, double _momentum, double _leakCoef, void (*_getData)(Vector*, Vector*)) {

    layers = _layers;

    learnRate = _learnRate;

    momentum = _momentum;

    leakCoef = _leakCoef;

    getData = _getData;

    //allocate mem
    A.push_back(new Vector(nodeCount[0]));

    aGrad.push_back(new Vector(nodeCount[0]));

    Y = new Vector(nodeCount[layers]);

    costBuffer = new double[nodeCount[layers]];

    for (int l = 1; l <= layers; l++) {

        A.push_back(new Vector(nodeCount[l]));

        B.push_back(new Vector(nodeCount[l]));

        W.push_back(new Matrix(nodeCount[l - 1], nodeCount[l]));

        aGrad.push_back(new Vector(nodeCount[l]));

        bGrad.push_back(new Vector(nodeCount[l]));

        bGradM.push_back(new Vector(nodeCount[l]));

        wGrad.push_back(new Matrix(nodeCount[l - 1], nodeCount[l]));

        wGradM.push_back(new Matrix(nodeCount[l - 1], nodeCount[l]));

        W[l - 1]->randomize(-0.03, 0.03);

    }

}

void NeuralNetwork::run() {

    getData(A[0], Y);

    A[0]->column();

    for (int l = 0; l < layers; l++) {

        A[l + 1]->column();

        A[l + 1]->multiplySet(A[l], W[l]);

        A[l + 1]->add(B[l]);

        lReLU(A[l + 1]);

    }

}

void NeuralNetwork::train(int itrs, int N) {

    //compute cost and deriv of cost with respect to final activation layer
    for (int i = 0; i < itrs; i++) {

        cost = 0;

        for (int n = 0; n < N; n++) {

            run();

            backProp(n == 0);

        }

        applyGradient(N);

        cost /= N;

    }

}

void NeuralNetwork::backProp(bool firstItr) {

    aGrad[layers]->sub(A[layers], Y);

    aGrad[layers]->copyTo(costBuffer);

    for (int i = 0; i < aGrad[layers]->length; i++)

        cost += costBuffer[i] * costBuffer[i] * 0.5;

    //Back Propagation
    for (int l = layers - 1; l >= 0; l--) {

        //calculate derivative of A dZ
        lReLU_Deriv(A[l + 1], aGrad[l + 1]);

        A[l]->row();

        aGrad[l + 1]->column();

        //multiply activations of layer k with dA/dZ to compute weight gradient
        if (firstItr)

            wGrad[l]->multiplySet(A[l], aGrad[l + 1]);

        else

            wGrad[l]->multiplyAdd(A[l], aGrad[l + 1]);

        aGrad[l]->row();

        aGrad[l + 1]->row();

        aGrad[l]->multiplySet(W[l], aGrad[l + 1]);

        bGrad[l]->add(aGrad[l + 1]);

    }

}

void NeuralNetwork::applyGradient(int N) {

    double scalar = -learnRate / N;

    for (int l = 0; l < layers; l++) {

        //apply learning rate to gradient
        wGrad[l]->multiplyScalar(scalar);

        bGrad[l]->multiplyScalar(scalar);

        //add gradient term to previous gradient term
        wGradM[l]->add(wGrad[l]);

        bGradM[l]->add(bGrad[l]);

        //add negative gradient to weights and biases
        W[l]->add(wGradM[l]);

        B[l]->add(bGradM[l]);

        //multiply momentum scalar
        wGradM[l]->multiplyScalar(momentum);

        bGradM[l]->multiplyScalar(momentum);

        bGrad[l]->fill(0);

    }

}

void NeuralNetwork::lReLU(Vector* A) {

    d_lReLU <<< A->gridDim, A->blockDim >>> (A->d_self, leakCoef, A->xD, A->yD, A->size);

}


void NeuralNetwork::lReLU_Deriv(Vector* A, Vector* B) {

    d_lReLU_Deriv <<< A->gridDim, A->blockDim >>> (A->d_self, B->d_self, leakCoef, A->xD, A->yD, A->size);

}

void NeuralNetwork::printOutput() {

    A[layers]->print();

}
