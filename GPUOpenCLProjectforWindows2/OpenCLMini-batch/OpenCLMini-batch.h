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

// Macros for OpenCL versions
#define OPENCL_VERSION_1_2  1.2f
#define OPENCL_VERSION_2_0  2.0f

// Suppress a compiler warning about undefined CL_TARGET_OPENCL_VERSION
// Khronos ICD supports only latest OpenCL version
#define CL_TARGET_OPENCL_VERSION 220

// Suppress a compiler warning about 'clCreateCommandQueue': was declared deprecated
// for OpenCL 1.2
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include "CL\cl.h"
#include "utils.h"
//#include "helperCl.h"

//for perf. counters
#include <Windows.h>

using namespace std::chrono;

#pragma once

struct ocl_args_d_t
{
    ocl_args_d_t();
    ~ocl_args_d_t();

    // Regular OpenCL objects:
    cl_context       context;           // hold the context handler
    cl_device_id     device;            // hold the selected device handler
    cl_command_queue commandQueue;      // hold the commands-queue handler
    cl_program       program;           // hold the program handler
    cl_program       programSimple;           // hold the program handler for naive GPU
    cl_kernel        kernel;            // hold the kernel handler
    float            platformVersion;   // hold the OpenCL platform version (default 1.2)
    float            deviceVersion;     // hold the OpenCL device version (default. 1.2)
    float            compilerVersion;   // hold the device OpenCL C version (default. 1.2)

    // Objects that are specific for algorithm implemented in this sample
    cl_mem           srcA;              // hold first source buffer
    cl_mem           srcB;              // hold second source buffer
    cl_mem           dstMem;            // hold destination buffer
};

bool CheckPreferredPlatformMatch(cl_platform_id platform, const char* preferredPlatform);

cl_platform_id FindOpenCLPlatform(const char* preferredPlatform, cl_device_type deviceType);


cl_platform_id GetFirstPlatform();

int PrintDeviceIDs(cl_platform_id platformId);

int GetPlatformAndDeviceVersion(cl_platform_id platformId, ocl_args_d_t* ocl);


void mGenerateMatrices(cl_float* inputArray, cl_uint height, cl_uint width);

int SetupOpenCL(ocl_args_d_t* ocl, cl_device_type deviceType);

int CreateAndBuildProgram(ocl_args_d_t* ocl);

cl_uint mSetKernelArguments(ocl_args_d_t* ocl, cl_mem* matrixD, cl_uint mDim, cl_uint pDim, cl_uint nDim, cl_float learningRate, cl_uint kernel);

cl_uint mExecuteMultiplyKernelCustom(ocl_args_d_t* ocl, cl_uint mDim, cl_uint nDim);

cl_uint executeMultiplyKernel(ocl_args_d_t* ocl, const size_t global[2], const size_t local[2]);

cl_uint mExecuteMultiplyKernel(ocl_args_d_t* ocl, cl_uint mDim, cl_uint nDim);


cl_float MSECostFunction(cl_float* correctOutput, cl_float* networkOutput, cl_uint batchSize);

cl_int listalldevices();

void ListDevices(cl_platform_id pid);


cl_uint initializeparamsClassifier(ocl_args_d_t* ocl, cl_mem** buffersWeightsArray, cl_mem** buffersOutsArray, cl_mem** buffersDeltasArray, cl_mem* bufferInputArray, cl_float** costs,
    int dimensions[], int batchSize, int iterations, int layers, cl_float** correctOutput, uchar** dataset, uchar* labels);

cl_uint forwardpassClassifier(ocl_args_d_t* ocl, cl_mem* buffersWeightsArray, cl_mem* buffersBiasesArray, cl_mem* buffersOutsArray, cl_mem* bufferInputArray,
    int dimensions[], int* ActivationFunctions, cl_kernel* activationFunctionKernels, cl_kernel* activationFunctionKernelsSimple, int batchSize, int layers);

cl_uint backpropClassifier(ocl_args_d_t* ocl, cl_mem* buffersWeightsArray, cl_mem* buffersBiasesArray, cl_mem* buffersOutsArray, cl_mem* buffersDeltasArray, cl_mem* bufferInputArray, int dimensions[],
    int* ActivationFunctions, cl_kernel* activationFunctionDeltaKernels, cl_kernel* activationFunctionDeltaKernelsSimple, cl_float* correctOutput, cl_float* costs, cl_float learning_rate, int iter, int batchSize, int layers, int classes);

cl_uint backpropClassifier2(ocl_args_d_t* ocl, cl_mem* buffersWeightsArray, cl_mem* buffersBiasesArray, cl_mem* buffersOutsArray, cl_mem* buffersDeltasArray, cl_mem* bufferInputArray, int dimensions[],
    int* ActivationFunctions, cl_kernel* activationFunctionDeltaKernels, cl_kernel* activationFunctionDeltaKernelsSimple, cl_float* correctOutput, cl_float* costs, cl_float learning_rate, int iter, int batchSize, int layers, int classes);

cl_uint backpropClassifier3(ocl_args_d_t* ocl, cl_mem* buffersWeightsArray, cl_mem* buffersBiasesArray, cl_mem* buffersOutsArray, cl_mem* buffersDeltasArray, cl_mem* bufferInputArray, int dimensions[],
    int* ActivationFunctions, cl_kernel* activationFunctionDeltaKernels, cl_kernel* activationFunctionDeltaKernelsSimple, cl_float* correctOutput, cl_float* costs, cl_float learning_rate, int iter, int batchSize, int layers, int classes);

cl_uint testingClassifier(ocl_args_d_t* ocl, cl_mem* buffersWeightsArray, cl_mem* buffersBiasesArray, cl_mem* buffersOutsArray, cl_mem* bufferInputArray, int dimensions[],
    int* ActivationFunctions, cl_kernel* activationFunctionKernels, cl_kernel* activationFunctionKernelsSimple, int layers, int classes, uchar** valDataset, uchar* valLabels, int numValImages);

uchar** read_mnist_images(std::string full_path, int& number_of_images, int& image_size);

uchar* read_mnist_labels(std::string full_path, int& number_of_labels);

cl_uint minibatchGD(ocl_args_d_t* ocl, int dimensions[], int* activationFunctions, cl_kernel* activationFunctionKernels, cl_kernel* activationFunctionKernelsSimple,
    cl_kernel* activationFunctionDeltaKernels, cl_kernel* activationFunctionDeltaKernelsSimple, int batchSize, int layers, int classes, int epochs,
    uchar** dataset, uchar* labels, int numTrainImages, uchar** valDataset, uchar* valLabels, int numValImages);

cl_uint minibatchGD2(ocl_args_d_t* ocl, int dimensions[], int* activationFunctions, cl_kernel* activationFunctionKernels, cl_kernel* activationFunctionKernelsSimple,
    cl_kernel* activationFunctionDeltaKernels, cl_kernel* activationFunctionDeltaKernelsSimple, int batchSize, int layers, int classes, int epochs,
    uchar** dataset, uchar* labels, int numTrainImages, uchar** valDataset, uchar* valLabels, int numValImages);

cl_int kernelCorrectnessTesting(ocl_args_d_t* ocl, char** activationFunctionKernelNames, char** activationFunctionDeltaKernelNames, cl_kernel* activationFunctionKernels,
    cl_kernel* activationFunctionDeltaKernels, cl_kernel* activationFunctionKernelsSimple, cl_kernel* activationFunctionDeltaKernelsSimple,
    const size_t global[2], const size_t local[2], int numActivationFunctions, int mDim, int pDim, int nDim);

long long kernelLatencyTestingAuxiliary(ocl_args_d_t* ocl, cl_kernel* activationFunctionKernels,
    const size_t global[2], const size_t local[2], int mDim, int pDim, int nDim, int iterations, int typeOfKernel);

void kernelLatencyTesting(ocl_args_d_t* ocl, cl_kernel* activationFunctionKernel, int iterations, int WGS, int TW, int typeOfKernel);