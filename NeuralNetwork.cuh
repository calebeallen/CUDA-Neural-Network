
#ifndef NEURAL_NETWORK
#define NEURAL_NETWORK

#include <vector>
#include "Linear.cuh"

class NeuralNetwork {

private:

    std::vector<Vector*> A, aGrad, B, bGrad, bGradM;

    std::vector<Matrix*> W, wGrad, wGradM;

    Vector* Y;

    int layers;

    void (*getData)(Vector*, Vector*);

    double* costBuffer, learnRate, momentum, leakCoef;

    void backProp(bool);

    void applyGradient(int);

    void lReLU(Vector*);

    void lReLU_Deriv(Vector*, Vector*);

public:

    double cost;

    NeuralNetwork(int, int*, double, double, double, void (*)(Vector*, Vector*));

    void run();

    void train(int, int);

    void printOutput();

};

#endif