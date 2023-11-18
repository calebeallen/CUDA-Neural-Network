
#include "NeuralNetwork.cuh"
#include <iostream>
#include <ctime>

using namespace std;

int seed;

int const layers = 3;

int const itrs = 100;

int const outputs = 10;

int const inputs = 1000;

int nodeCount[4] = { inputs, 500, 500, outputs };

double input[inputs];

double y[outputs];

void createTrainingData( int s ) {

    int offset = rand() % 10;

    for (int i = 0; i < inputs; i++) {

        input[offset % inputs] = (double)(i % ((s + 1) * 2)) + (((double)(rand()) / (RAND_MAX + 1)) * (0.09 - 0.0001) + 0.0001) / 10;

        offset++;

    }

    for (int i = 0; i < outputs; i++)

        y[i] = 0;

    y[s] = 1;

}

void getData(Vector* inputLayer, Vector* Y) {

    createTrainingData( seed == -1 ? rand() % outputs : seed );
    
    inputLayer->copy(input);

    Y->copy(y);

}

int main() {

    NeuralNetwork NN(layers, nodeCount, 0.01, 0.9, 0.01, &getData);

    int option, trainAmt;

    srand(time(NULL));

    cudaSetDevice(0);

    //Simple command prompt interface
    while (true) {

        cout << "\nTest(0) Train(1) End(2): ";

        cin >> option;

        switch (option) {

            //Test network
            case 0:

                cout << "\nSeed: ";

                cin >> seed;

                NN.run();

                cout << "\nOUTPUT LAYER";

                NN.printOutput();

                break;

            //Train network
            case 1:

                cout << "\nTraining iterations: ";

                cin >> trainAmt;

                seed = -1;

                NN.train(trainAmt, itrs);

                cout << "\nNetwork Cost: " << NN.cost << "\n";

                break;

        }

    }

}

