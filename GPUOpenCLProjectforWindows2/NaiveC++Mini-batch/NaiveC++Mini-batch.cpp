#include <random>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <tchar.h>
#include <memory.h>
#include <vector>
#include <cmath>
#include <ctime>
#include <chrono> 

#include <fstream>

typedef unsigned char uchar;

//for perf. counters
#include <Windows.h>

using namespace std::chrono;

#pragma once

float AccuracyFunction(float* correctOutput, int* choices, int batchSize) {

    int total = 0;
    for (int x = 0; x < batchSize; ++x) {
        if ((int)correctOutput[x] == choices[x]) {
            total += 1;
        }
        //std::cout << "network output for " << x << "th example is " << networkOutput[batchSize * correctClass + x] << '\n';
        //std::cout << "correct class for " << x << "th example is " << correctOutput[x] << '\n';
    }
    std::cout << "Accuracy is " << ((float)total / (float)batchSize) << '\n';
    return ((float)total / (float)batchSize);
}

//Performing AxB, x is normal matrix multiplication
//A = weights, B = outputs for layer l-1, C = outputs for l, 
//where l is the current layer visited in this iteration of the forwardpassCpp loop
void multiplyIdKernelCpp(float* matrixA, float* matrixB, float* matrixC,
    const int mDim, const int pDim, const int nDim, float* biasesArray) {
    int idx;
    float temp;
    for (int i = 0; i < mDim; i++) {
        for (int j = 0; j < nDim; j++) {
            idx = i * nDim + j;
            temp = 0.0f;
            for (int k = 0; k < pDim; k++) {
                temp += matrixA[i * pDim + k] * matrixB[j + k * nDim];
            }
            matrixC[idx] = temp + biasesArray[i];
            //std::cout << "le IDENTITY issa " << matrixC[idx] << '\n';
        }
    }
}

//Performing C = AxB and then elementwise sigmoid() on C
//x is normal matrix multiplication
//A = weights, B = outputs for layer l-1, C = outputs for l, 
//where l is the current layer visited in this iteration of the forwardpassCpp loop
void multiplySigmoidKernelCpp(float* matrixA, float* matrixB, float* matrixC,
    const int mDim, const int pDim, const int nDim, float* biasesArray) {
    int idx;
    float temp;
    for (int i = 0; i < mDim; i++) {
        for (int j = 0; j < nDim; j++) {
            idx = i * nDim + j;
            temp = 0.0f;
            for (int k = 0; k < pDim; k++) {
                temp += matrixA[i * pDim + k] * matrixB[j + k * nDim];
            }
            matrixC[idx] = (tanh((temp + biasesArray[i]) / 2) + 1) / 2; //expressed sig as tanh because sig is not implemented in cmath
        }
    }
}

//Performing AxB, and then elementwise tanh() on C
//x is normal matrix multiplication
//A = weights, B = outputs for layer l-1, C = outputs for l, 
//where l is the current layer visited in this iteration of the forwardpassCpp loop
void multiplyTanhKernelCpp(float* matrixA, float* matrixB, float* matrixC,
    const int mDim, const int pDim, const int nDim, float* biasesArray) {
    int idx;
    float temp;
    for (int i = 0; i < mDim; i++) {
        for (int j = 0; j < nDim; j++) {
            idx = i * nDim + j;
            temp = 0.0f;
            for (int k = 0; k < pDim; k++) {
                temp += matrixA[i * pDim + k] * matrixB[j + k * nDim];
            }
            matrixC[idx] = tanh(temp + biasesArray[i]);
        }
    }
}

//Performing AxB, and then elementwise ReLU() on C
//x is normal matrix multiplication
//A = weights, B = outputs for layer l-1, C = outputs for l,
//where l is the current layer visited in this iteration of the forwardpassCpp loop
void multiplyReLUKernelCpp(float* matrixA, float* matrixB, float* matrixC,
    const int mDim, const int pDim, const int nDim, float* biasesArray) {
    int idx;
    float temp;
    //printinn(matrixA, matrixB, matrixC, mDim, pDim, nDim);
    for (int i = 0; i < mDim; i++) {
        for (int j = 0; j < nDim; j++) {
            idx = i * nDim + j;
            temp = 0.0f;
            for (int k = 0; k < pDim; k++) {
                temp += matrixA[i * pDim + k] * matrixB[j + k * nDim];
            }
            //std::cout << "le output pre relu issa " << temp << '\n';
            matrixC[idx] = fmax(temp + biasesArray[i], 0.0f);
            //std::cout << "le output post relu issa " << matrixC[idx] << '\n';
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////BP1&BP2//////////////////////////////////////
//Performing A.TxB, x is normal matrix multiplication
//A = weights, B = deltas for layer l, C = deltas for l-1, 
//where l is the current layer visited in this iteration of the backpropCpp loop
void multiplyDeltasId(float* matrixA, float* matrixB, float* matrixC,
    const int mDim, const int pDim, const int nDim) {
    int idx;
    float temp;
    for (int i = 0; i < mDim; i++) {
        for (int j = 0; j < nDim; j++) {
            idx = i * nDim + j;
            temp = 0.0f;
            for (int k = 0; k < pDim; k++) {
                temp += matrixA[i + k * mDim] * matrixB[j + k * nDim];
            }
            matrixC[idx] = fmin(fmax(temp, -0.005f), 0.005f);
        }
    }
}

//Performing A.TxB and then elementwise grad of sigmoid() on C
//x is normal matrix multiplication
//A = weights, B = deltas for layer l, C = deltas for l-1, D = outputs for l-1
//where l is the current layer visited in this iteration of the backpropCpp loop
void multiplyDeltasSigmoid(float* matrixA, float* matrixB, float* matrixC,
    const int mDim, const int pDim, const int nDim, float* matrixD) {
    int idx;
    float temp;
    for (int i = 0; i < mDim; i++) {
        for (int j = 0; j < nDim; j++) {
            idx = i * nDim + j;
            temp = 0.0f;
            for (int k = 0; k < pDim; k++) {
                temp += matrixA[i + k * mDim] * matrixB[j + k * nDim];
            }
            matrixC[idx] = fmin(fmax(temp * matrixD[idx] * (1.0f - matrixD[idx]), -0.005f), 0.005f);
        }
    }
}

//Performing A.TxB and then elementwise grad of tanh() on C
//x is normal matrix multiplication
//A = weights, B = deltas for layer l, C = deltas for l-1, D = outputs for l-1
//where l is the current layer visited in this iteration of the backpropCpp loop
void multiplyDeltasTanh(float* matrixA, float* matrixB, float* matrixC,
    const int mDim, const int pDim, const int nDim, float* matrixD) {
    int idx;
    float temp;
    for (int i = 0; i < mDim; i++) {
        for (int j = 0; j < nDim; j++) {
            idx = i * nDim + j;
            temp = 0.0f;
            for (int k = 0; k < pDim; k++) {
                temp += matrixA[i + k * mDim] * matrixB[j + k * nDim];
            }
            matrixC[idx] = fmin(fmax(temp * (1.0f - pow(matrixD[idx], 2)), -0.005f), 0.005f);
        }
    }
}

//Performing A.TxB and then elementwise grad of ReLU() on C
//x is normal matrix multiplication
//A = weights, B = deltas for layer l, C = deltas for l-1, D = outputs for l-1
//where l is the current layer visited in this iteration of the backpropCpp loop
void multiplyDeltasReLU(float* matrixA, float* matrixB, float* matrixC,
    const int mDim, const int pDim, const int nDim, float* matrixD) {
    int idx;
    float temp;
    for (int i = 0; i < mDim; i++) {
        for (int j = 0; j < nDim; j++) {
            idx = i * nDim + j;
            temp = 0.0f;
            for (int k = 0; k < pDim; k++) {
                temp += matrixA[i + k * mDim] * matrixB[j + k * nDim];
            }
            matrixC[idx] = fmin(fmax(temp * (matrixD[idx] > 0.0 ? 1.0 : 0.0), -0.005f), 0.005f);
        }
    }
}
//Performing C' = AxB.T and then C = C - offset*C'
//x is normal matrix multiplication, * is element wise
//A = deltas, B = outputs, C = weights, offset = learning rate
void updateWeights(float* matrixA, float* matrixB, float* matrixC,
    const int mDim, const int pDim, const int nDim, float* biasesArray, const float offset)
{
    std::cout << "IN THE UPDATESSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS" << '\n';
    float temp;
    int idx1, idx2;
    for (int i = 0; i < mDim; i++) {
        idx1 = i * pDim;
        for (int j = 0; j < nDim; j++) {
            temp = 0.0f;
            idx2 = j * pDim;
            for (int k = 0; k < pDim; k++) {
                temp += matrixA[idx1 + k] * matrixB[idx2 + k];
                //std::cout << "delta val is " << matrixA[idx1 + k] << '\n';
                //std::cout << "output val is " << matrixB[idx2 + k] << '\n';
            }
            temp = temp / (float)pDim;
            matrixC[i * nDim + j] = matrixC[i * nDim + j] - offset * temp;
            //std::cout << matrixC[i * nDim + j]<<'\n';
            //std::cout << "after update is " << matrixC[i * nDim + j] << '\n';
        }
        temp = 0.0f;
        for (int j = 0; j < pDim; ++j) {
            temp += matrixA[i * pDim + j];
        }
        temp = temp / (float)pDim;
        biasesArray[i] = biasesArray[i] - offset * temp;
        //std::cout << biasesArray[i] << '\n';
    }
    return;
}
//////////////////////////////////////////////////////////////////////////////////END OF KERNELS///////////////////////////////////////////////////////////////////////////////////////////


void mGenerateMatrices(float* inputArray, int height, int width)
{
    float temp = 0;

    std::random_device rd;
    srand((unsigned int)rd());
    float interval = 2.0;
    float lowerLimit = -1.0;

    //random initialization of input
    int array_size = height * width;
    for (int i = 0; i < array_size; ++i)
    {
        temp = lowerLimit + (float(rand()) / float((RAND_MAX)) * interval);
        inputArray[i] = temp;
        //std::cout << temp << " ";
        if ((i + 1) % width == 0) {
            //std::cout << '\n';
        }
    }
}

void forwardpassClassifierCpp(float** weightArrays, float** biasArrays, float** outputArrays, float* inputArray, int dimensions[], int* activationFunctions, int batchSize, int layers) {
    std::cout << "In forwardprop \n";
    float* srcA, * srcB;
    float* dstMem = inputArray;
    int mDim, pDim, nDim = batchSize, kernel;
    auto start = high_resolution_clock::now();
    auto elapsed = std::chrono::high_resolution_clock::now() - start;
    long long microseconds;
    for (int x = 0; x < layers; ++x) {

        mDim = dimensions[x + 1];
        pDim = dimensions[x];

        srcA = weightArrays[x];
        srcB = dstMem;
        dstMem = outputArrays[x];

        kernel = activationFunctions[x];
        switch (kernel) {
        case 0:


            std::cout << "ENTERING Cpp kernel calculation" << '\n';
            start = high_resolution_clock::now();
            multiplyIdKernelCpp(srcA, srcB, dstMem, mDim, pDim, nDim, biasArrays[x]);
            elapsed = std::chrono::high_resolution_clock::now() - start;
            microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
            std::cout << microseconds << " microseconds for training \n";
            system("pause");
            break;
        case 1:
            multiplySigmoidKernelCpp(srcA, srcB, dstMem, mDim, pDim, nDim, biasArrays[x]);
            break;
        case 2:
            multiplyTanhKernelCpp(srcA, srcB, dstMem, mDim, pDim, nDim, biasArrays[x]);
            break;
        case 3:
            multiplyReLUKernelCpp(srcA, srcB, dstMem, mDim, pDim, nDim, biasArrays[x]);
            break;
        }
    }
}

void backpropClassifierCpp(float** weightArrays, float** biasArrays, float** outputArrays, float** deltaArrays, float* inputArray, int* dimensions,
    float* groundTruthArray, int* activationFunctions, float* costs, float learning_rate, int iter, int batchSize, int layers, int classes) {

    //std::cout << "In backprop for iter " << iter << " \n";

    int optimizedSize = ((sizeof(float) * batchSize * classes - 1) / 64 + 1) * 64;
    float* softmaxOutputs = (float*)_aligned_malloc(optimizedSize, 4096);

    optimizedSize = ((sizeof(int) * batchSize - 1) / 64 + 1) * 64;
    int* choices = (int*)_aligned_malloc(optimizedSize, 4096);

    float* preSoftmaxOutputs = outputArrays[layers - 1];

    int idx, correctClass;
    float maxval, temp;
    for (int i = 0; i < batchSize; ++i) {
        //std::cout << "Entering calculation for batch element " << i << "\n";
        idx = 0;
        temp = 0.0f;
        maxval = -FLT_MAX;
        correctClass = (int)groundTruthArray[i];
        for (int j = 0; j < classes; ++j) {
            //std::cout << "presoftmax for " << j << " th class and batch item " << i << " is " << preSoftmaxOutputs[j * batchSize + i] << '\n';
            if (preSoftmaxOutputs[j * batchSize + i] > maxval) {
                idx = j;
                maxval = preSoftmaxOutputs[j * batchSize + i];
            }
        }
        //system("pause");
        choices[i] = idx;
        //std::cout << "Correct class is " << correctClass << '\n';
        //std::cout << "predicted class is " << idx << '\n';
        for (int j = 0; j < classes; ++j) {
            preSoftmaxOutputs[j * batchSize + i] = preSoftmaxOutputs[j * batchSize + i] - maxval;
            softmaxOutputs[j * batchSize + i] = exp(preSoftmaxOutputs[j * batchSize + i]);
            temp += exp(preSoftmaxOutputs[j * batchSize + i]);
        }
        for (int j = 0; j < classes; ++j) {
            softmaxOutputs[j * batchSize + i] = softmaxOutputs[j * batchSize + i] / temp;
        }
    }

    //These delta calculation formulas correspond to a CEL cost function
    for (int i = 0; i < batchSize; ++i) {
        correctClass = (int)groundTruthArray[i];
        for (int j = 0; j < classes; ++j) {
            if (j == correctClass) {
                deltaArrays[layers - 1][j * batchSize + i] = softmaxOutputs[j * batchSize + i] - 1.0f;
            }
            else {
                deltaArrays[layers - 1][j * batchSize + i] = softmaxOutputs[j * batchSize + i];
            }
        }
    }

    costs[iter] = AccuracyFunction(groundTruthArray, choices, batchSize);

    //non-output deltas calculation loop
    float* srcA, * srcB, * dstMem;
    dstMem = deltaArrays[layers - 1];
    int mDim, pDim, nDim = batchSize, kernel;
    auto start = high_resolution_clock::now();
    auto elapsed = std::chrono::high_resolution_clock::now() - start;
    long long microseconds;
    for (int x = layers - 1; x > 0; --x) {

        mDim = dimensions[x];
        pDim = dimensions[x + 1];
        srcA = weightArrays[x];
        srcB = dstMem;
        dstMem = deltaArrays[x - 1];

        kernel = activationFunctions[x];
        switch (kernel) {
        case 0:

            std::cout << "ENTERING Cpp delta kernel calculation" << '\n';
            start = high_resolution_clock::now();
            multiplyDeltasId(srcA, srcB, dstMem, mDim, pDim, nDim);
            elapsed = std::chrono::high_resolution_clock::now() - start;
            microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
            std::cout << microseconds << " microseconds for training \n";
            system("pause");
            break;
        case 1:
            multiplyDeltasSigmoid(srcA, srcB, dstMem, mDim, pDim, nDim, outputArrays[x - 1]);
            break;
        case 2:
            multiplyDeltasTanh(srcA, srcB, dstMem, mDim, pDim, nDim, outputArrays[x - 1]);
            break;
        case 3:
            multiplyDeltasReLU(srcA, srcB, dstMem, mDim, pDim, nDim, outputArrays[x - 1]);
            break;
        }
    }

    //perform weight updates now. We can potentially parallelize this fully even across all network layers
    //but for now it's only across the weights of each layer and then sequentially across layers
    pDim = batchSize;
    for (int x = layers - 1; x >= 0; --x) {
        //std::cout << "I'm in iteration " << x << " of the weight update loop \n";
        mDim = dimensions[x + 1];
        nDim = dimensions[x];
        srcA = deltaArrays[x];
        dstMem = weightArrays[x];
        if (x != 0) {
            srcB = outputArrays[x - 1];
        }
        else {
            srcB = inputArray;
        }
        std::cout << "ENTERING Cpp weight update kernel calculation" << '\n';
        std::cout << "Dimensions are" << mDim << " " << pDim << " " << nDim << '\n';
        start = high_resolution_clock::now();
        updateWeights(srcA, srcB, dstMem, mDim, pDim, nDim, biasArrays[x], learning_rate);
        elapsed = std::chrono::high_resolution_clock::now() - start;
        microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
        std::cout << microseconds << " microseconds for training \n";
        system("pause");
    }
}

int testingClassifierCpp(float** weightArrays, float** biasArrays, float** outputArrays, int dimensions[],
    int* activationFunctions, int layers, int classes, uchar** valDataset, uchar* valLabels, int numValImages) {

    std::cout << "In validation \n";

    int optimizedSize = ((sizeof(float) * dimensions[0] * numValImages - 1) / 64 + 1) * 64;
    float* inputArray = (float*)_aligned_malloc(optimizedSize, 4096); //array of network input

    for (int i = 0; i < numValImages; ++i) {
        for (int j = 0; j < dimensions[0]; ++j) {
            inputArray[j * numValImages + i] = (float)valDataset[i][j];
        }
    }

    optimizedSize = ((sizeof(float) * numValImages - 1) / 64 + 1) * 64;
    float* groundTruthArray = (float*)_aligned_malloc(optimizedSize, 4096); //array of network ground truth

    for (int i = 0; i < numValImages; ++i) {
        groundTruthArray[i] = (float)valLabels[i];
    }


    int mDim, pDim, nDim = numValImages;
    for (int x = 0; x < layers; ++x) {
        mDim = dimensions[x + 1];
        pDim = dimensions[x];

        optimizedSize = ((sizeof(float) * mDim * nDim - 1) / 64 + 1) * 64;
        outputArrays[x] = (float*)_aligned_malloc(optimizedSize, 4096);
    }

    float* srcA, * srcB;
    float* dstMem = inputArray;
    int kernel;
    for (int x = 0; x < layers; ++x) {

        mDim = dimensions[x + 1];
        pDim = dimensions[x];

        srcA = weightArrays[x];
        srcB = dstMem;
        dstMem = outputArrays[x];

        kernel = activationFunctions[x];
        switch (kernel) {
        case 0:
            multiplyIdKernelCpp(srcA, srcB, dstMem, mDim, pDim, nDim, biasArrays[x]);
            break;
        case 1:
            multiplySigmoidKernelCpp(srcA, srcB, dstMem, mDim, pDim, nDim, biasArrays[x]);
            break;
        case 2:
            multiplyTanhKernelCpp(srcA, srcB, dstMem, mDim, pDim, nDim, biasArrays[x]);
            break;
        case 3:
            multiplyReLUKernelCpp(srcA, srcB, dstMem, mDim, pDim, nDim, biasArrays[x]);
            break;
        }
    }

    optimizedSize = ((sizeof(float) * numValImages * classes - 1) / 64 + 1) * 64;
    float* preSoftmaxOutputs = (float*)_aligned_malloc(optimizedSize, 4096);
    float* softmaxOutputs = (float*)_aligned_malloc(optimizedSize, 4096);

    optimizedSize = ((sizeof(int) * numValImages - 1) / 64 + 1) * 64;
    int* choices = (int*)_aligned_malloc(optimizedSize, 4096);


    int idx, correctClass;
    float maxval, temp;
    for (int i = 0; i < numValImages; ++i) {
        idx = 0;
        temp = 0.0f;
        maxval = -FLT_MAX;
        correctClass = (int)groundTruthArray[i];
        for (int j = 0; j < classes; ++j) {
            if (dstMem[j * numValImages + i] > maxval) {
                idx = j;
                maxval = dstMem[j * numValImages + i];
            }
        }
        choices[i] = idx;
        for (int j = 0; j < classes; ++j) {
            dstMem[j * numValImages + i] = dstMem[j * numValImages + i] - maxval;
        }
    }

    float accuracy = AccuracyFunction(groundTruthArray, choices, numValImages);
    std::cout << "Validation set accuracy is " << accuracy << '\n';
    return 0;
}

int minibatchGDCpp(int dimensions[], int* activationFunctions, int batchSize, int layers, int classes, int epochs, uchar** dataset, uchar* labels,
    int numTrainImages, uchar** valDataset, uchar* valLabels, int numValImages) {

    float** weightsAr;
    float** weightArrays, ** biasArrays, ** outputArrays, ** deltaArrays, ** inputArrays, ** groundTruthArrays;
    float* costs;

    std::cout << "Num of val images is " << numValImages << '\n';

    int itersPerEpoch = (numTrainImages - 1) / batchSize + 1;
    int iterations = epochs * (itersPerEpoch - 1);

    int optimizedSize = ((sizeof(float*) * layers - 1) / 64 + 1) * 64;
    weightArrays = (float**)_aligned_malloc(optimizedSize, 4096);//array of memory objects, where each memory object is a buffer of weights between layers
    biasArrays = (float**)_aligned_malloc(optimizedSize, 4096); //array of memory objects, where each memory object is a buffer of biases for some layer
    outputArrays = (float**)_aligned_malloc(optimizedSize, 4096); //array of memory objects, where each memory object is a buffer image of outputs of layers
    deltaArrays = (float**)_aligned_malloc(optimizedSize, 4096); //array of memory objects, where each memory object is an image of deltas of layers

    optimizedSize = ((sizeof(float*) * (itersPerEpoch - 1) - 1) / 64 + 1) * 64;
    inputArrays = (float**)_aligned_malloc(optimizedSize, 4096);
    groundTruthArrays = (float**)_aligned_malloc(optimizedSize, 4096); //array of network grount truth

    int optimizedSizeIn = ((sizeof(float) * dimensions[0] * batchSize - 1) / 64 + 1) * 64;
    int optimizedSizeOut = ((sizeof(float) * batchSize - 1) / 64 + 1) * 64;

    for (int iter = 0; iter < itersPerEpoch - 1; ++iter) {
        inputArrays[iter] = (float*)_aligned_malloc(optimizedSizeIn, 4096);
        groundTruthArrays[iter] = (float*)_aligned_malloc(optimizedSizeOut, 4096);
        for (int i = 0; i < batchSize; ++i) {
            for (int j = 0; j < dimensions[0]; ++j) {
                inputArrays[iter][j * batchSize + i] = (float)dataset[i + batchSize * iter][j];
            }
            groundTruthArrays[iter][i] = labels[i + batchSize * iter];
        }
    }

    int mDim, pDim;
    for (int x = 0; x < layers; ++x) {
        mDim = dimensions[x + 1];
        pDim = dimensions[x];

        optimizedSize = ((sizeof(float) * mDim * pDim - 1) / 64 + 1) * 64;
        weightArrays[x] = (float*)_aligned_malloc(optimizedSize, 4096);
        std::cout << "Weights of layer " << x << " are: \n";
        mGenerateMatrices(weightArrays[x], mDim, pDim);

        optimizedSize = ((sizeof(float) * mDim - 1) / 64 + 1) * 64;
        biasArrays[x] = (float*)_aligned_malloc(optimizedSize, 4096);
        std::cout << "Biases of layer " << x << " are: \n";
        mGenerateMatrices(biasArrays[x], mDim, 1);

        optimizedSize = ((sizeof(float) * mDim * batchSize - 1) / 64 + 1) * 64;
        outputArrays[x] = (float*)_aligned_malloc(optimizedSize, 4096);

        optimizedSize = ((sizeof(float) * mDim * batchSize - 1) / 64 + 1) * 64;
        deltaArrays[x] = (float*)_aligned_malloc(optimizedSize, 4096);
    }

    //initializing weights, outputs and delta buffers
    int optimizedSizeCosts = ((sizeof(float) * epochs * (itersPerEpoch - 1) - 1) / 64 + 1) * 64;
    costs = (float*)_aligned_malloc(optimizedSizeCosts, 4096);

    float learning_rate = 0.0008;
    float temptot;
    std::cout << "iters per epoch is " << (itersPerEpoch - 1) << '\n';
    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::cout << "epoch " << epoch << " hereeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee \n";
        learning_rate *= 0.92;
        for (int i = 0; i < itersPerEpoch - 1; ++i) {
            std::cout << "ENTERING Cpp FORWARD" << "\n";
            forwardpassClassifierCpp(weightArrays, biasArrays, outputArrays, inputArrays[i], dimensions, activationFunctions, batchSize, layers);
            std::cout << "ENTERING Cpp BACKWARD" << "\n";
            backpropClassifierCpp(weightArrays, biasArrays, outputArrays, deltaArrays, inputArrays[i], dimensions, groundTruthArrays[i], activationFunctions, costs, learning_rate, (epoch * (itersPerEpoch - 1) + i), batchSize, layers, classes);
        }
        temptot = 0.0f;
        for (int i = 0; i < itersPerEpoch - 1; ++i) {
            temptot += costs[epoch * (itersPerEpoch - 1) + i];
        }
        std::cout << "average accuracy for epoch " << epoch << " is " << temptot / (itersPerEpoch - 1) << '\n';
    }

    for (int i = 0; i < iterations; i++) {
        std::cout << costs[i] << '\n';
    }

    //For reasonably sized validation datasets, a single pass with a wide matrix is enough
    for (int x = 0; x < layers; ++x) {
        _aligned_free(outputArrays[x]);
        _aligned_free(deltaArrays[x]);
    }
    _aligned_free(deltaArrays);
    _aligned_free(costs);

    testingClassifierCpp(weightArrays, biasArrays, outputArrays, dimensions, activationFunctions, layers, classes, valDataset, valLabels, numValImages);

    for (int x = 0; x < layers; ++x) {
        //std::cout << "releasing obj num" << x;
        _aligned_free(weightArrays[x]);
        _aligned_free(biasArrays[x]);
    }
    for (int x = 0; x < itersPerEpoch - 1; ++x) {
        //std::cout << "releasing obj num" << x;
        _aligned_free(inputArrays[x]);
        _aligned_free(groundTruthArrays[x]);
    }
    std::cout << "Done with everything \n";
    system("pause");
    _aligned_free(weightArrays);
    _aligned_free(biasArrays);
    _aligned_free(outputArrays);
    _aligned_free(inputArrays);
    return 0;
}
