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

float AccuracyFunction(float* correctOutput, int* choices, int batchSize);


void multiplyIdKernelCpp(float* matrixA, float* matrixB, float* matrixC,
    const int mDim, const int pDim, const int nDim, float* biasesArray);

void multiplySigmoidKernelCpp(float* matrixA, float* matrixB, float* matrixC,
    const int mDim, const int pDim, const int nDim, float* biasesArray);


void multiplyTanhKernelCpp(float* matrixA, float* matrixB, float* matrixC,
    const int mDim, const int pDim, const int nDim, float* biasesArray);


void multiplyReLUKernelCpp(float* matrixA, float* matrixB, float* matrixC,
    const int mDim, const int pDim, const int nDim, float* biasesArray);

void multiplyDeltasId(float* matrixA, float* matrixB, float* matrixC,
    const int mDim, const int pDim, const int nDim);

void multiplyDeltasSigmoid(float* matrixA, float* matrixB, float* matrixC,
    const int mDim, const int pDim, const int nDim, float* matrixD);

void multiplyDeltasTanh(float* matrixA, float* matrixB, float* matrixC,
    const int mDim, const int pDim, const int nDim, float* matrixD);

void multiplyDeltasReLU(float* matrixA, float* matrixB, float* matrixC,
    const int mDim, const int pDim, const int nDim, float* matrixD);

void updateWeights(float* matrixA, float* matrixB, float* matrixC,
    const int mDim, const int pDim, const int nDim, float* biasesArray, const float offset);

void forwardpassClassifierCpp(float** weightArrays, float** biasArrays, float** outputArrays, float* inputArray, int dimensions[], int* activationFunctions, int batchSize, int layers);

void backpropClassifierCpp(float** weightArrays, float** biasArrays, float** outputArrays, float** deltaArrays, float* inputArray, int* dimensions,
    float* groundTruthArray, int* activationFunctions, float* costs, float learning_rate, int iter, int batchSize, int layers, int classes);

int testingClassifierCpp(float** weightArrays, float** biasArrays, float** outputArrays, int dimensions[],
    int* activationFunctions, int layers, int classes, uchar** valDataset, uchar* valLabels, int numValImages);

int minibatchGDCpp(int dimensions[], int* activationFunctions, int batchSize, int layers, int classes, int epochs, uchar** dataset, uchar* labels,
    int numTrainImages, uchar** valDataset, uchar* valLabels, int numValImages);