/*****************************************************************************
 * Copyright (c) 2013-2016 Intel Corporation
 * All rights reserved.
 *
 * WARRANTY DISCLAIMER
 *
 * THESE MATERIALS ARE PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL INTEL OR ITS
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THESE
 * MATERIALS, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Intel Corporation is the author of the Materials, and requests that all
 * problem reports or change requests be submitted to it directly
 *****************************************************************************/

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

//for perf. counters
#include <Windows.h>

#define dim 512  //MAY REMOVE LATERR

/* This function helps to create informative messages in
 * case when OpenCL errors occur. It returns a string
 * representation for an OpenCL error code.
 * (E.g. "CL_DEVICE_NOT_FOUND" instead of just -1.)
 */
const char* TranslateOpenCLError(cl_int errorCode)
{
    switch(errorCode)
    {
    case CL_SUCCESS:                            return "CL_SUCCESS";
    case CL_DEVICE_NOT_FOUND:                   return "CL_DEVICE_NOT_FOUND";
    case CL_DEVICE_NOT_AVAILABLE:               return "CL_DEVICE_NOT_AVAILABLE";
    case CL_COMPILER_NOT_AVAILABLE:             return "CL_COMPILER_NOT_AVAILABLE";
    case CL_MEM_OBJECT_ALLOCATION_FAILURE:      return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case CL_OUT_OF_RESOURCES:                   return "CL_OUT_OF_RESOURCES";
    case CL_OUT_OF_HOST_MEMORY:                 return "CL_OUT_OF_HOST_MEMORY";
    case CL_PROFILING_INFO_NOT_AVAILABLE:       return "CL_PROFILING_INFO_NOT_AVAILABLE";
    case CL_MEM_COPY_OVERLAP:                   return "CL_MEM_COPY_OVERLAP";
    case CL_IMAGE_FORMAT_MISMATCH:              return "CL_IMAGE_FORMAT_MISMATCH";
    case CL_IMAGE_FORMAT_NOT_SUPPORTED:         return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    case CL_BUILD_PROGRAM_FAILURE:              return "CL_BUILD_PROGRAM_FAILURE";
    case CL_MAP_FAILURE:                        return "CL_MAP_FAILURE";
    case CL_MISALIGNED_SUB_BUFFER_OFFSET:       return "CL_MISALIGNED_SUB_BUFFER_OFFSET";                          //-13
    case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:    return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";   //-14
    case CL_COMPILE_PROGRAM_FAILURE:            return "CL_COMPILE_PROGRAM_FAILURE";                               //-15
    case CL_LINKER_NOT_AVAILABLE:               return "CL_LINKER_NOT_AVAILABLE";                                  //-16
    case CL_LINK_PROGRAM_FAILURE:               return "CL_LINK_PROGRAM_FAILURE";                                  //-17
    case CL_DEVICE_PARTITION_FAILED:            return "CL_DEVICE_PARTITION_FAILED";                               //-18
    case CL_KERNEL_ARG_INFO_NOT_AVAILABLE:      return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";                         //-19
    case CL_INVALID_VALUE:                      return "CL_INVALID_VALUE";
    case CL_INVALID_DEVICE_TYPE:                return "CL_INVALID_DEVICE_TYPE";
    case CL_INVALID_PLATFORM:                   return "CL_INVALID_PLATFORM";
    case CL_INVALID_DEVICE:                     return "CL_INVALID_DEVICE";
    case CL_INVALID_CONTEXT:                    return "CL_INVALID_CONTEXT";
    case CL_INVALID_QUEUE_PROPERTIES:           return "CL_INVALID_QUEUE_PROPERTIES";
    case CL_INVALID_COMMAND_QUEUE:              return "CL_INVALID_COMMAND_QUEUE";
    case CL_INVALID_HOST_PTR:                   return "CL_INVALID_HOST_PTR";
    case CL_INVALID_MEM_OBJECT:                 return "CL_INVALID_MEM_OBJECT";
    case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:    return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case CL_INVALID_IMAGE_SIZE:                 return "CL_INVALID_IMAGE_SIZE";
    case CL_INVALID_SAMPLER:                    return "CL_INVALID_SAMPLER";
    case CL_INVALID_BINARY:                     return "CL_INVALID_BINARY";
    case CL_INVALID_BUILD_OPTIONS:              return "CL_INVALID_BUILD_OPTIONS";
    case CL_INVALID_PROGRAM:                    return "CL_INVALID_PROGRAM";
    case CL_INVALID_PROGRAM_EXECUTABLE:         return "CL_INVALID_PROGRAM_EXECUTABLE";
    case CL_INVALID_KERNEL_NAME:                return "CL_INVALID_KERNEL_NAME";
    case CL_INVALID_KERNEL_DEFINITION:          return "CL_INVALID_KERNEL_DEFINITION";
    case CL_INVALID_KERNEL:                     return "CL_INVALID_KERNEL";
    case CL_INVALID_ARG_INDEX:                  return "CL_INVALID_ARG_INDEX";
    case CL_INVALID_ARG_VALUE:                  return "CL_INVALID_ARG_VALUE";
    case CL_INVALID_ARG_SIZE:                   return "CL_INVALID_ARG_SIZE";
    case CL_INVALID_KERNEL_ARGS:                return "CL_INVALID_KERNEL_ARGS";
    case CL_INVALID_WORK_DIMENSION:             return "CL_INVALID_WORK_DIMENSION";
    case CL_INVALID_WORK_GROUP_SIZE:            return "CL_INVALID_WORK_GROUP_SIZE";
    case CL_INVALID_WORK_ITEM_SIZE:             return "CL_INVALID_WORK_ITEM_SIZE";
    case CL_INVALID_GLOBAL_OFFSET:              return "CL_INVALID_GLOBAL_OFFSET";
    case CL_INVALID_EVENT_WAIT_LIST:            return "CL_INVALID_EVENT_WAIT_LIST";
    case CL_INVALID_EVENT:                      return "CL_INVALID_EVENT";
    case CL_INVALID_OPERATION:                  return "CL_INVALID_OPERATION";
    case CL_INVALID_GL_OBJECT:                  return "CL_INVALID_GL_OBJECT";
    case CL_INVALID_BUFFER_SIZE:                return "CL_INVALID_BUFFER_SIZE";
    case CL_INVALID_MIP_LEVEL:                  return "CL_INVALID_MIP_LEVEL";
    case CL_INVALID_GLOBAL_WORK_SIZE:           return "CL_INVALID_GLOBAL_WORK_SIZE";                           //-63
    case CL_INVALID_PROPERTY:                   return "CL_INVALID_PROPERTY";                                   //-64
    case CL_INVALID_IMAGE_DESCRIPTOR:           return "CL_INVALID_IMAGE_DESCRIPTOR";                           //-65
    case CL_INVALID_COMPILER_OPTIONS:           return "CL_INVALID_COMPILER_OPTIONS";                           //-66
    case CL_INVALID_LINKER_OPTIONS:             return "CL_INVALID_LINKER_OPTIONS";                             //-67
    case CL_INVALID_DEVICE_PARTITION_COUNT:     return "CL_INVALID_DEVICE_PARTITION_COUNT";                     //-68
//    case CL_INVALID_PIPE_SIZE:                  return "CL_INVALID_PIPE_SIZE";                                  //-69
//    case CL_INVALID_DEVICE_QUEUE:               return "CL_INVALID_DEVICE_QUEUE";                               //-70    

    default:
        return "UNKNOWN ERROR CODE";
    }
}


/* Convenient container for all OpenCL specific objects used in the sample
 *
 * It consists of two parts:
 *   - regular OpenCL objects which are used in almost each normal OpenCL applications
 *   - several OpenCL objects that are specific for this particular sample
 *
 * You collect all these objects in one structure for utility purposes
 * only, there is no OpenCL specific here: just to avoid global variables
 * and make passing all these arguments in functions easier.
 */
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

ocl_args_d_t::ocl_args_d_t() :
    context(NULL),
    device(NULL),
    commandQueue(NULL),
    program(NULL),
    programSimple(NULL),
    kernel(NULL),
    platformVersion(OPENCL_VERSION_1_2),
    deviceVersion(OPENCL_VERSION_1_2),
    compilerVersion(OPENCL_VERSION_1_2),
    srcA(NULL),
    srcB(NULL),
    dstMem(NULL)
{
}
/*
 * destructor - called only once
 * Release all OpenCL objects
 * This is a regular sequence of calls to deallocate all created OpenCL resources in bootstrapOpenCL.
 *
 * You may want to call these deallocation procedures in the middle of your application execution
 * (not at the end) if you don't further need OpenCL runtime.
 * You may want to do that in order to free some memory, for example,
 * or recreate OpenCL objects with different parameters.
 *
 */
    ocl_args_d_t::~ocl_args_d_t()
{
    cl_int err = CL_SUCCESS;

    if (kernel)
    {
        err = clReleaseKernel(kernel);
        if (CL_SUCCESS != err)
        {
            LogError("Error: clReleaseKernel returned '%s'.\n", TranslateOpenCLError(err));
        }
    }
    if (program)
    {
        err = clReleaseProgram(program);
        if (CL_SUCCESS != err)
        {
            LogError("Error: clReleaseProgram returned '%s'.\n", TranslateOpenCLError(err));
        }
    }
    if (programSimple)
    {
        err = clReleaseProgram(programSimple);
        if (CL_SUCCESS != err)
        {
            LogError("Error: clReleaseProgram returned '%s'.\n", TranslateOpenCLError(err));
        }
    }
    /*if (srcA)
    {
        err = clReleaseMemObject(srcA);
        if (CL_SUCCESS != err)
        {
            LogError("Error: clReleaseMemObject returned '%s'.\n", TranslateOpenCLError(err));
        }
    }
    if (srcB)
    {
        err = clReleaseMemObject(srcB);
        if (CL_SUCCESS != err)
        {
            LogError("Error: clReleaseMemObject returned '%s'.\n", TranslateOpenCLError(err));
        }
    }
    if (dstMem)
    {
        err = clReleaseMemObject(dstMem);
        if (CL_SUCCESS != err)
        {
            LogError("Error: clReleaseMemObject returned '%s'.\n", TranslateOpenCLError(err));
        }
    }*/
    if (commandQueue)
    {
        err = clReleaseCommandQueue(commandQueue);
        if (CL_SUCCESS != err)
        {
            LogError("Error: clReleaseCommandQueue returned '%s'.\n", TranslateOpenCLError(err));
        }
    }
    if (device)
    {
        err = clReleaseDevice(device);
        if (CL_SUCCESS != err)
        {
            LogError("Error: clReleaseDevice returned '%s'.\n", TranslateOpenCLError(err));
        }
    }
    if (context)
    {
        err = clReleaseContext(context);
        if (CL_SUCCESS != err)
        {
            LogError("Error: clReleaseContext returned '%s'.\n", TranslateOpenCLError(err));
        }
    }

    /*
     * Note there is no procedure to deallocate platform
     * because it was not created at the startup,
     * but just queried from OpenCL runtime.
     */
}


/*
 * Check whether an OpenCL platform is the required platform
 * (based on the platform's name)
 */
bool CheckPreferredPlatformMatch(cl_platform_id platform, const char* preferredPlatform)
{
    size_t stringLength = 0;
    cl_int err = CL_SUCCESS;
    bool match = false;

    // In order to read the platform's name, we first read the platform's name string length (param_value is NULL).
    // The value returned in stringLength
    err = clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, NULL, &stringLength);
    if (CL_SUCCESS != err)
    {
        LogError("Error: clGetPlatformInfo() to get CL_PLATFORM_NAME length returned '%s'.\n", TranslateOpenCLError(err));
        return false;
    }

    // Now, that we know the platform's name string length, we can allocate enough space before read it
    std::vector<char> platformName(stringLength);

    // Read the platform's name string
    // The read value returned in platformName
    err = clGetPlatformInfo(platform, CL_PLATFORM_NAME, stringLength, &platformName[0], NULL);
    if (CL_SUCCESS != err)
    {
        LogError("Error: clGetplatform_ids() to get CL_PLATFORM_NAME returned %s.\n", TranslateOpenCLError(err));
        return false;
    }
    
    // Now check if the platform's name is the required one
    if (strstr(&platformName[0], preferredPlatform) != 0)
    {
        // The checked platform is the one we're looking for
        LogInfo("Platform: %s\n", &platformName[0]);
        match = true;
    }

    return match;
}


/*
 * Find and return the preferred OpenCL platform
 * In case that preferredPlatform is NULL, the ID of the first discovered platform will be returned
 */
cl_platform_id FindOpenCLPlatform(const char* preferredPlatform, cl_device_type deviceType)
{
    cl_uint numPlatforms = 0;
    cl_int err = CL_SUCCESS;

    // Get (in numPlatforms) the number of OpenCL platforms available
    // No platform ID will be return, since platforms is NULL
    err = clGetPlatformIDs(0, NULL, &numPlatforms);
    if (CL_SUCCESS != err)
    {
        LogError("Error: clGetplatform_ids() to get num platforms returned %s.\n", TranslateOpenCLError(err));
        return NULL;
    }
    LogInfo("Number of available platforms: %u\n", numPlatforms);

    if (0 == numPlatforms)
    {
        LogError("Error: No platforms found!\n");
        return NULL;
    }

    std::vector<cl_platform_id> platforms(numPlatforms);

    // Now, obtains a list of numPlatforms OpenCL platforms available
    // The list of platforms available will be returned in platforms
    err = clGetPlatformIDs(numPlatforms, &platforms[0], NULL);
    if (CL_SUCCESS != err)
    {
        LogError("Error: clGetplatform_ids() to get platforms returned %s.\n", TranslateOpenCLError(err));
        return NULL;
    }

    // Check if one of the available platform matches the preferred requirements
    for (cl_uint i = 0; i < numPlatforms; i++)
    {
        bool match = true;
        cl_uint numDevices = 0;

        // If the preferredPlatform is not NULL then check if platforms[i] is the required one
        // Otherwise, continue the check with platforms[i]
        if ((NULL != preferredPlatform) && (strlen(preferredPlatform) > 0))
        {
            // In case we're looking for a specific platform
            match = CheckPreferredPlatformMatch(platforms[i], preferredPlatform);
        }

        // match is true if the platform's name is the required one or don't care (NULL)
        if (match)
        {
            // Obtains the number of deviceType devices available on platform
            // When the function failed we expect numDevices to be zero.
            // We ignore the function return value since a non-zero error code
            // could happen if this platform doesn't support the specified device type.
            err = clGetDeviceIDs(platforms[i], deviceType, 0, NULL, &numDevices);
            if (CL_SUCCESS != err)
            {
                LogInfo("   Required device was not found on this platform.\n");
            }

            if (0 != numDevices)
            {
                // There is at list one device that answer the requirements
                LogInfo("   Required device was found.\n");
                return platforms[i];
            }
        }
    }

    LogError("Error: Required device was not found on any platform.\n");
    return NULL;
}


cl_platform_id GetFirstPlatform()
{
    cl_uint numPlatforms = 0;
    cl_int err = CL_SUCCESS;

    // Get (in numPlatforms) the number of OpenCL platforms available
    // No platform ID will be return, since platforms is NULL
    err = clGetPlatformIDs(0, NULL, &numPlatforms);
    if (CL_SUCCESS != err)
    {
        LogError("Error: clGetplatform_ids() to get num platforms returned %s.\n", TranslateOpenCLError(err));
        return NULL;
    }
    LogInfo("Number of available platforms: %u\n", numPlatforms);

    if (0 == numPlatforms)
    {
        LogError("Error: No platforms found!\n");
        return NULL;
    }

    std::vector<cl_platform_id> platforms(numPlatforms);

    // Now, obtains a list of numPlatforms OpenCL platforms available
    // The list of platforms available will be returned in platforms
    err = clGetPlatformIDs(numPlatforms, &platforms[0], NULL);
    if (CL_SUCCESS != err)
    {
        LogError("Error: clGetplatform_ids() to get platforms returned %s.\n", TranslateOpenCLError(err));
        return NULL;
    }

    return platforms[0];
}

//My code, goes over all devices and prints their features
int PrintDeviceIDs(cl_platform_id platformId)
{
    cl_int err = CL_SUCCESS;

    // Read the platform's version string length (param_value is NULL).
    // The value returned in stringLength
    cl_uint numEntries = 0;
    err = clGetDeviceIDs(platformId, CL_DEVICE_TYPE_GPU, 0, NULL, &numEntries);
    if (CL_SUCCESS != err)
        return err;

    // Now, that we know the platform's version string length, we can allocate enough space before read it
    cl_device_id* deviceIds= (cl_device_id*)malloc(sizeof(cl_device_id)*numEntries);

    // Read the platform's version string
    // The read value returned in platformVersion
    err = clGetDeviceIDs(platformId, CL_DEVICE_TYPE_GPU, numEntries, deviceIds, NULL);
    if (CL_SUCCESS != err)
        return err;

    cl_device_id temp;
    cl_device_type deviceType;
    cl_uint computeUnits;
    cl_uint nDims;
    size_t wGSize;
    std::cout << "THIS MANY DEVICES "<<numEntries;
    for (int i = 0; i < numEntries; i++) {
        temp = deviceIds[i];

        err = clGetDeviceInfo(temp, CL_DEVICE_TYPE, sizeof(cl_device_type), &deviceType, NULL);
        std::cout << "deviceId: "<<deviceIds[i]<<'\n';
        err = clGetDeviceInfo(temp, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_device_type), &computeUnits, NULL);
        std::cout << "computeUnits: "<<computeUnits << '\n';
        err = clGetDeviceInfo(temp, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint), &nDims, NULL);
        std::cout <<"nDims: "<< nDims << '\n';

        //cl_uint optimizedSize = ((sizeof(size_t) * nDims - 1) / 64 + 1) * 64;
        //size_t* wGSizeArray = (size_t*)_aligned_malloc(optimizedSize, 4096);
        size_t* wGSizeArray = (size_t*)malloc(sizeof(size_t)*nDims);
        err = clGetDeviceInfo(temp, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(size_t)*nDims, wGSizeArray, NULL);
        for (int i=0; i<nDims; i++)
            std::cout <<"Workgroup dim "<<i<<" is "<< wGSizeArray[i] << '\n';
        err = clGetDeviceInfo(temp, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &wGSize, NULL);
        std::cout << "WG size: "<<wGSize << '\n';
        free(wGSizeArray);
    }
}


/*
 * This function read the OpenCL platform and device versions
 * (using clGetxxxInfo API) and stores it in the ocl structure.
 * Later it will enable us to support both OpenCL 1.2 and 2.0 platforms and devices
 * in the same program.
 */
int GetPlatformAndDeviceVersion (cl_platform_id platformId, ocl_args_d_t *ocl)
{
    cl_int err = CL_SUCCESS;

    // Read the platform's version string length (param_value is NULL).
    // The value returned in stringLength
    size_t stringLength = 0;
    err = clGetPlatformInfo(platformId, CL_PLATFORM_VERSION, 0, NULL, &stringLength);
    if (CL_SUCCESS != err)
    {
        LogError("Error: clGetPlatformInfo() to get CL_PLATFORM_VERSION length returned '%s'.\n", TranslateOpenCLError(err));
        return err;
    }

    // Now, that we know the platform's version string length, we can allocate enough space before read it
    std::vector<char> platformVersion(stringLength);

    // Read the platform's version string
    // The read value returned in platformVersion
    err = clGetPlatformInfo(platformId, CL_PLATFORM_VERSION, stringLength, &platformVersion[0], NULL);
    if (CL_SUCCESS != err)
    {
        LogError("Error: clGetplatform_ids() to get CL_PLATFORM_VERSION returned %s.\n", TranslateOpenCLError(err));
        return err;
    }

    if (strstr(&platformVersion[0], "OpenCL 2.0") != NULL)
    {
        ocl->platformVersion = OPENCL_VERSION_2_0;
    }

    // Read the device's version string length (param_value is NULL).
    err = clGetDeviceInfo(ocl->device, CL_DEVICE_VERSION, 0, NULL, &stringLength);
    if (CL_SUCCESS != err)
    {
        LogError("Error: clGetDeviceInfo() to get CL_DEVICE_VERSION length returned '%s'.\n", TranslateOpenCLError(err));
        return err;
    }

    // Now, that we know the device's version string length, we can allocate enough space before read it
    std::vector<char> deviceVersion(stringLength);

    // Read the device's version string
    // The read value returned in deviceVersion
    err = clGetDeviceInfo(ocl->device, CL_DEVICE_VERSION, stringLength, &deviceVersion[0], NULL);
    if (CL_SUCCESS != err)
    {
        LogError("Error: clGetDeviceInfo() to get CL_DEVICE_VERSION returned %s.\n", TranslateOpenCLError(err));
        return err;
    }

    if (strstr(&deviceVersion[0], "OpenCL 2.0") != NULL)
    {
        ocl->deviceVersion = OPENCL_VERSION_2_0;
    }

    // Read the device's OpenCL C version string length (param_value is NULL).
    err = clGetDeviceInfo(ocl->device, CL_DEVICE_OPENCL_C_VERSION, 0, NULL, &stringLength);
    if (CL_SUCCESS != err)
    {
        LogError("Error: clGetDeviceInfo() to get CL_DEVICE_OPENCL_C_VERSION length returned '%s'.\n", TranslateOpenCLError(err));
        return err;
    }

    // Now, that we know the device's OpenCL C version string length, we can allocate enough space before read it
    std::vector<char> compilerVersion(stringLength);

    // Read the device's OpenCL C version string
    // The read value returned in compilerVersion
    err = clGetDeviceInfo(ocl->device, CL_DEVICE_OPENCL_C_VERSION, stringLength, &compilerVersion[0], NULL);
    if (CL_SUCCESS != err)
    {
        LogError("Error: clGetDeviceInfo() to get CL_DEVICE_OPENCL_C_VERSION returned %s.\n", TranslateOpenCLError(err));
        return err;
    }

    else if (strstr(&compilerVersion[0], "OpenCL C 2.0") != NULL)
    {
        ocl->compilerVersion = OPENCL_VERSION_2_0;
    }

    return err;
}


/*
 * Generate random value for input buffers
 */
/*
void generateInput(cl_int* inputArray, cl_uint arrayWidth, cl_uint arrayHeight)
{
    srand(12345);

    // random initialization of input
    cl_uint array_size = arrayWidth * arrayHeight;
    for (cl_uint i = 0; i < array_size; ++i)
    {
        inputArray[i] = rand();
    }
}*/
 

void mGenerateMatrices(cl_float* inputArray, cl_uint height, cl_uint width)
{
    //srand(12345);
    cl_float temp = 0;

    /*std::random_device rd; // obtain a random number from hardware
    std::mt19937 gen(rd()); // seed the generator
    std::uniform_int_distribution<> distr(-5, 5); // define the range
    temp = distr(gen);*/
    std::random_device rd;
    srand((unsigned int)rd());
    cl_float interval = 2.0;
    cl_float lowerLimit = -1.0;

    //random initialization of input
    cl_uint array_size = height * width;
    for (cl_uint i = 0; i < array_size; ++i)
    {
        temp = lowerLimit + (cl_float(rand()) / cl_float((RAND_MAX)) * interval);
        inputArray[i] = temp;
        //std::cout << temp << " ";
        if ((i + 1) % width == 0) {
            //std::cout << '\n';
        }
    }

    std::cout << std::string(10, '\n');
}

/*
 * This function picks/creates necessary OpenCL objects which are needed.
 * The objects are:
 * OpenCL platform, device, context, and command queue.
 *
 * All these steps are needed to be performed once in a regular OpenCL application.
 * This happens before actual compute kernels calls are performed.
 *
 * For convenience, in this application you store all those basic OpenCL objects in structure ocl_args_d_t,
 * so this function populates fields of this structure, which is passed as parameter ocl.
 * Please, consider reviewing the fields before going further.
 * The structure definition is right in the beginning of this file.
 */
int SetupOpenCL(ocl_args_d_t *ocl, cl_device_type deviceType)
{
    // The following variable stores return codes for all OpenCL calls.
    cl_int err = CL_SUCCESS;

    // Query for all available OpenCL platforms on the system
    // Here you enumerate all platforms and pick one which name has preferredPlatform as a sub-string
    cl_platform_id platformId = FindOpenCLPlatform("Intel", deviceType);
    if (NULL == platformId)
    {
        LogError("Error: Failed to find OpenCL platform.\n");
        return CL_INVALID_VALUE;
    }

    // Create context with device of specified type.
    // Required device type is passed as function argument deviceType.
    // So you may use this function to create context for any CPU or GPU OpenCL device.
    // The creation is synchronized (pfn_notify is NULL) and NULL user_data
    cl_context_properties contextProperties[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)platformId, 0};
    ocl->context = clCreateContextFromType(contextProperties, deviceType, NULL, NULL, &err);
    if ((CL_SUCCESS != err) || (NULL == ocl->context))
    {
        LogError("Couldn't create a context, clCreateContextFromType() returned '%s'.\n", TranslateOpenCLError(err));
        return err;
    }

    // Query for OpenCL device which was used for context creation
    err = clGetContextInfo(ocl->context, CL_CONTEXT_DEVICES, sizeof(cl_device_id), &ocl->device, NULL);
    if (CL_SUCCESS != err)
    {
        LogError("Error: clGetContextInfo() to get list of devices returned %s.\n", TranslateOpenCLError(err));
        return err;
    }

    // Read the OpenCL platform's version and the device OpenCL and OpenCL C versions
    GetPlatformAndDeviceVersion(platformId, ocl);

    // Create command queue.
    // OpenCL kernels are enqueued for execution to a particular device through special objects called command queues.
    // Command queue guarantees some ordering between calls and other OpenCL commands.
    // Here you create a simple in-order OpenCL command queue that doesn't allow execution of two kernels in parallel on a target device.
#ifdef CL_VERSION_2_0
    if (OPENCL_VERSION_2_0 == ocl->deviceVersion)
    {
        const cl_command_queue_properties properties[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
        ocl->commandQueue = clCreateCommandQueueWithProperties(ocl->context, ocl->device, properties, &err);
    } 
    else {
        // default behavior: OpenCL 1.2
        cl_command_queue_properties properties = CL_QUEUE_PROFILING_ENABLE;
        ocl->commandQueue = clCreateCommandQueue(ocl->context, ocl->device, properties, &err);
    } 
#else
    // default behavior: OpenCL 1.2
    cl_command_queue_properties properties = CL_QUEUE_PROFILING_ENABLE;
    ocl->commandQueue = clCreateCommandQueue(ocl->context, ocl->device, properties, &err);
#endif
    if (CL_SUCCESS != err)
    {
        LogError("Error: clCreateCommandQueue() returned %s.\n", TranslateOpenCLError(err));
        return err;
    }
    
    return CL_SUCCESS;
}


/* 
 * Create and build OpenCL program from its source code
 */
int CreateAndBuildProgram(ocl_args_d_t *ocl)
{
    cl_int err = CL_SUCCESS;

    // Upload the OpenCL C source code from the input file to source
    // The size of the C program is returned in sourceSize
    char* source = NULL;
    size_t srcSize = 0;
    err = ReadSourceFromFile("kernel1.cl", &source, &srcSize);
    if (CL_SUCCESS != err)
    {
        LogError("Error: ReadSourceFromFile returned %s.\n", TranslateOpenCLError(err));
        goto Finish;
    }
    char* sourceSimple = NULL;
    size_t srcSizeSimple = 0;
    err = ReadSourceFromFile("kernel1.cl", &sourceSimple, &srcSizeSimple);
    if (CL_SUCCESS != err)
    {
        LogError("Error: ReadSourceFromFile returned %s.\n", TranslateOpenCLError(err));
        goto Finish;
    }

    // And now after you obtained a regular C string call clCreateProgramWithSource to create OpenCL program object.
    ocl->program = clCreateProgramWithSource(ocl->context, 1, (const char**)&source, &srcSizeSimple, &err);
    if (CL_SUCCESS != err)
    {
        LogError("Error: clCreateProgramWithSource returned %s.\n", TranslateOpenCLError(err));
        goto Finish;
    }
    ocl->programSimple = clCreateProgramWithSource(ocl->context, 1, (const char**)&sourceSimple, &srcSizeSimple, &err);
    if (CL_SUCCESS != err)
    {
        LogError("Error: clCreateProgramWithSource returned %s.\n", TranslateOpenCLError(err));
        goto Finish;
    }

    // Build the program
    // During creation a program is not built. You need to explicitly call build function.
    // Here you just use create-build sequence,
    // but there are also other possibilities when program consist of several parts,
    // some of which are libraries, and you may want to consider using clCompileProgram and clLinkProgram as
    // alternatives.
    err = clBuildProgram(ocl->program, 1, &ocl->device, "", NULL, NULL);
    if (CL_SUCCESS != err)
    {
        LogError("Error: clBuildProgram() for source program returned %s.\n", TranslateOpenCLError(err));

        // In case of error print the build log to the standard output
        // First check the size of the log
        // Then allocate the memory and obtain the log from the program
        if (err == CL_BUILD_PROGRAM_FAILURE)
        {
            size_t log_size = 0;
            clGetProgramBuildInfo(ocl->program, ocl->device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

            std::vector<char> build_log(log_size);
            clGetProgramBuildInfo(ocl->program, ocl->device, CL_PROGRAM_BUILD_LOG, log_size, &build_log[0], NULL);

            LogError("Error happened during the build of OpenCL program.\nBuild log:%s", &build_log[0]);
        }
    }
    err = clBuildProgram(ocl->programSimple, 1, &ocl->device, "", NULL, NULL);
    if (CL_SUCCESS != err)
    {
        LogError("Error: clBuildProgram() for source program returned %s.\n", TranslateOpenCLError(err));

        // In case of error print the build log to the standard output
        // First check the size of the log
        // Then allocate the memory and obtain the log from the program
        if (err == CL_BUILD_PROGRAM_FAILURE)
        {
            size_t log_size = 0;
            clGetProgramBuildInfo(ocl->programSimple, ocl->device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

            std::vector<char> build_log(log_size);
            clGetProgramBuildInfo(ocl->programSimple, ocl->device, CL_PROGRAM_BUILD_LOG, log_size, &build_log[0], NULL);

            LogError("Error happened during the build of OpenCL program.\nBuild log:%s", &build_log[0]);
        }
    }
Finish:
    if (source)
    {
        delete[] source;
        source = NULL;
    }    
    if (sourceSimple)
    {
        delete[] sourceSimple;
        sourceSimple = NULL;
    }

    return err;
}

/*
 * Set kernel arguments
 */
cl_uint mSetKernelArguments(ocl_args_d_t *ocl, cl_mem* outputs, cl_uint mDim, cl_uint pDim, cl_uint nDim, cl_float learningRate, cl_uint kernel)
{
    cl_int err = CL_SUCCESS;

    err  =  clSetKernelArg(ocl->kernel, 0, sizeof(cl_mem), (void *)&ocl->srcA);
    if (CL_SUCCESS != err)
    {
        LogError("error: Failed to set argument srcA, returned %s\n", TranslateOpenCLError(err));
        return err;
    }

    err  = clSetKernelArg(ocl->kernel, 1, sizeof(cl_mem), (void *)&ocl->srcB);
    if (CL_SUCCESS != err)
    {
        LogError("Error: Failed to set argument srcB, returned %s\n", TranslateOpenCLError(err));
        return err;
    }

    err  = clSetKernelArg(ocl->kernel, 2, sizeof(cl_mem), (void *)&ocl->dstMem);
    if (CL_SUCCESS != err)
    {
        LogError("Error: Failed to set argument dstMem, returned %s\n", TranslateOpenCLError(err));
        return err;
    }

    err = clSetKernelArg(ocl->kernel, 3, sizeof(cl_uint), &mDim);
    if (CL_SUCCESS != err)
    {
       LogError("Error: Failed to set argument dstMem, returned %s\n", TranslateOpenCLError(err));
       return err;
    }

    err = clSetKernelArg(ocl->kernel, 4, sizeof(cl_uint), &pDim);
    if (CL_SUCCESS != err)
    {
        LogError("Error: Failed to set argument dstMem, returned %s\n", TranslateOpenCLError(err));
        return err;
    }

    err = clSetKernelArg(ocl->kernel, 5, sizeof(cl_uint), &nDim);
    if (CL_SUCCESS != err)
    {
        LogError("Error: Failed to set argument dstMem, returned %s\n", TranslateOpenCLError(err));
        return err;
    }

    if (kernel == 2) {
        err = clSetKernelArg(ocl->kernel, 6, sizeof(cl_mem), (void*)outputs);
        if (CL_SUCCESS != err)
        {
            LogError("Error: Failed to set argument dstMem, returned %s\n", TranslateOpenCLError(err));
            return err;
        }
    }

    if (kernel == 3) {//here it's the learning rate
        err = clSetKernelArg(ocl->kernel, 6, sizeof(cl_float), &learningRate);
        if (CL_SUCCESS != err)
        {
            LogError("Error: Failed to set argument dstMem, returned %s\n", TranslateOpenCLError(err));
            return err;
        }
    }

    return err;
}

cl_uint mExecuteMultiplyKernelCustom(ocl_args_d_t* ocl, const size_t* global, const size_t* local)
{
    cl_int err = CL_SUCCESS;

    // Define global iteration space for clEnqueueNDRangeKernel.

    // execute kernel
    err = clEnqueueNDRangeKernel(ocl->commandQueue, ocl->kernel, 2, NULL, global, local, 0, NULL, NULL);
    if (CL_SUCCESS != err)
    {
        LogError("Error: Failed to run kernel, return %s\n", TranslateOpenCLError(err));
        return err;
    }

    // Wait until the queued kernel is completed by the device
    err = clFinish(ocl->commandQueue);
    if (CL_SUCCESS != err)
    {
        LogError("Error: clFinish return %s\n", TranslateOpenCLError(err));
        return err;
    }

    return CL_SUCCESS;
}

cl_uint mExecuteMultiplyKernel(ocl_args_d_t* ocl, cl_uint mDim, cl_uint nDim)
{
    cl_int err = CL_SUCCESS;

    const int TS = 16;
    const size_t global[2] = { mDim, nDim};
    const size_t local[2] = { TS, TS };

    // execute kernel
    err = clEnqueueNDRangeKernel(ocl->commandQueue, ocl->kernel, 2, NULL, global, NULL, 0, NULL, NULL);
    if (CL_SUCCESS != err)
    {
        LogError("Error: Failed to run kernel, return %s\n", TranslateOpenCLError(err));
        return err;
    }

    // Wait until the queued kernel is completed by the device
    err = clFinish(ocl->commandQueue);
    if (CL_SUCCESS != err)
    {
        LogError("Error: clFinish return %s\n", TranslateOpenCLError(err));
        return err;
    }

    return CL_SUCCESS;
}

cl_uint mExecuteScalarMultiplyKernel(ocl_args_d_t* ocl, cl_uint mDim)
{
    cl_int err = CL_SUCCESS;

    // Define global iteration space for clEnqueueNDRangeKernel.
    size_t globalWorkSize[1] = {mDim};

    // execute kernel
    err = clEnqueueNDRangeKernel(ocl->commandQueue, ocl->kernel, 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);
    if (CL_SUCCESS != err)
    {
        LogError("Error: Failed to run kernel, return %s\n", TranslateOpenCLError(err));
        return err;
    }

    // Wait until the queued kernel is completed by the device
    err = clFinish(ocl->commandQueue);
    if (CL_SUCCESS != err)
    {
        LogError("Error: clFinish return %s\n", TranslateOpenCLError(err));
        return err;
    }

    return CL_SUCCESS;
}

cl_float MSECostFunction(cl_float* correctOutput, cl_float* networkOutput, cl_uint batchSize) {
    cl_float avg = 0.0;
    cl_float pointLoss;
    for (int x = 0; x < batchSize; ++x) {
        pointLoss = pow(networkOutput[x] - correctOutput[x],2);
        avg += pointLoss / (cl_float)batchSize; //divide here instead of in the end to avoid numerical overflow
        std::cout << "network output for "<<x<<"th example is "<< networkOutput[x] << '\n';
    }
    avg = avg / 2.0;
    return avg;
}

cl_float AccuracyFunction(cl_float* correctOutput, int* choices, cl_uint batchSize) {
    
    int total=0;
    for (int x = 0; x < batchSize; ++x) {
        if ((int)correctOutput[x] == choices[x] ) {
            total += 1;
        }
        //std::cout << "network output for " << x << "th example is " << networkOutput[batchSize * correctClass + x] << '\n';
        //std::cout << "correct class for " << x << "th example is " << correctOutput[x] << '\n';
    }
    std::cout << "Accuracy is "<< ((float)total / (float)batchSize)<<'\n';
    return ((float)total / (float)batchSize);
}

void mPrint2(ocl_args_d_t* ocl, cl_uint mDim, cl_uint pDim, cl_uint nDim, cl_uint whichKernel)
{
    std::cout << "\n \n IN MPRINT 2\n";
    cl_int err = CL_SUCCESS;

    cl_uint optimizedSize1 = ((sizeof(cl_float) * mDim * pDim - 1) / 64 + 1) * 64;
    cl_uint optimizedSize2 = ((sizeof(cl_float) * pDim * nDim - 1) / 64 + 1) * 64;
    cl_uint optimizedSize3 = ((sizeof(cl_float) * mDim * nDim - 1) / 64 + 1) * 64;
    cl_float* resultPtr1 = (cl_float*)_aligned_malloc(optimizedSize1, 4096);
    cl_float* resultPtr2 = (cl_float*)_aligned_malloc(optimizedSize2, 4096);
    cl_float* resultPtr3 = (cl_float*)_aligned_malloc(optimizedSize3, 4096);

    err = clEnqueueReadBuffer(ocl->commandQueue, ocl->srcA, true, 0, sizeof(cl_float) * mDim * pDim, resultPtr1, 0, NULL, NULL);
    if (CL_SUCCESS != err)
    {
        LogError("Error: clEnqueueReadBuffer returned %s\n", TranslateOpenCLError(err));
    }
    err = clEnqueueReadBuffer(ocl->commandQueue, ocl->srcB, true, 0, sizeof(cl_float) * pDim * nDim, resultPtr2, 0, NULL, NULL);
    if (CL_SUCCESS != err)
    {
        LogError("Error: clEnqueueReadBuffer returned %s\n", TranslateOpenCLError(err));
    }
    err = clEnqueueReadBuffer(ocl->commandQueue, ocl->dstMem, true, 0, sizeof(cl_float) * mDim * nDim, resultPtr3, 0, NULL, NULL);
    if (CL_SUCCESS != err)
    {
        LogError("Error: clEnqueueReadBuffer returned %s\n", TranslateOpenCLError(err));
    }
    err = clFinish(ocl->commandQueue);
    if (CL_SUCCESS != err)
    {
        LogError("Error: clEnqueueReadBuffer returned %s\n", TranslateOpenCLError(err));
    }

    cl_int ptr1Width = pDim;
    cl_int ptr2Width = nDim;
    cl_int ptr3Width = nDim;
    /*if (whichKernel == 2) {
        ptr1Width = mDim;
    }
    else if (whichKernel == 3) {
        ptr2Width = pDim;
    }*/

    unsigned int size = mDim * pDim;
    for (unsigned int i = 0; i < size; ++i)
    {
        std::cout << resultPtr1[i] << " ";
        if ((i + 1) % ptr1Width == 0) {
            std::cout << '\n';
        }
    }
    std::cout << std::string(10, '\n');

    size = pDim * nDim;
    for (unsigned int i = 0; i < size; ++i)
    {
        std::cout << resultPtr2[i] << " ";
        if ((i + 1) % ptr2Width == 0) {
            std::cout << '\n';
        }
    }
    std::cout << std::string(10, '\n');

    size = mDim * nDim;
    for (unsigned int i = 0; i < size; ++i)
    {
        std::cout << resultPtr3[i] << " ";
        if ((i + 1) % ptr3Width == 0) {
            std::cout << '\n';
        }
    }
    std::cout << std::string(10, '\n');

    _aligned_free(resultPtr1);
    _aligned_free(resultPtr2);
    _aligned_free(resultPtr3);
}


cl_int listalldevices() {

    int i, j;
    char* value;
    size_t valueSize;
    cl_uint platformCount;
    cl_platform_id* platforms;
    cl_uint deviceCount;
    cl_device_id* devices;
    cl_uint maxComputeUnits;

    // get all platforms
    clGetPlatformIDs(0, NULL, &platformCount);
    platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * platformCount);
    clGetPlatformIDs(platformCount, platforms, NULL);

    for (i = 0; i < platformCount; i++) {

        // get all devices
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 0, NULL, &deviceCount);
        devices = (cl_device_id*)malloc(sizeof(cl_device_id) * deviceCount);
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, deviceCount, devices, NULL);

        // for each device print critical attributes
        for (j = 0; j < deviceCount; j++) {

            // print device name
            clGetDeviceInfo(devices[j], CL_DEVICE_NAME, 0, NULL, &valueSize);
            value = (char*)malloc(valueSize);
            clGetDeviceInfo(devices[j], CL_DEVICE_NAME, valueSize, value, NULL);
            printf("%d. Device: %s\n", j + 1, value);
            free(value);

            // print hardware device version
            clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, 0, NULL, &valueSize);
            value = (char*)malloc(valueSize);
            clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, valueSize, value, NULL);
            printf(" %d.%d Hardware version: %s\n", j + 1, 1, value);
            free(value);

            // print software driver version
            clGetDeviceInfo(devices[j], CL_DRIVER_VERSION, 0, NULL, &valueSize);
            value = (char*)malloc(valueSize);
            clGetDeviceInfo(devices[j], CL_DRIVER_VERSION, valueSize, value, NULL);
            printf(" %d.%d Software version: %s\n", j + 1, 2, value);
            free(value);

            // print c version supported by compiler for device
            clGetDeviceInfo(devices[j], CL_DEVICE_OPENCL_C_VERSION, 0, NULL, &valueSize);
            value = (char*)malloc(valueSize);
            clGetDeviceInfo(devices[j], CL_DEVICE_OPENCL_C_VERSION, valueSize, value, NULL);
            printf(" %d.%d OpenCL C version: %s\n", j + 1, 3, value);
            free(value);

            // print parallel compute units
            clGetDeviceInfo(devices[j], CL_DEVICE_MAX_COMPUTE_UNITS,
                sizeof(maxComputeUnits), &maxComputeUnits, NULL);
            printf(" %d.%d Parallel compute units: %d\n", j + 1, 4, maxComputeUnits);

        }

        free(devices);

    }

    free(platforms);
    return 0;

}

void ListDevices(cl_platform_id pid) {

    size_t valueSize;
    char* value;
    cl_uint platformCount;
    cl_platform_id* platforms;
    cl_uint deviceCount;
    cl_device_id* devices;
    cl_uint maxComputeUnits;

    clGetDeviceIDs(pid, CL_DEVICE_TYPE_ALL, 0, NULL, &deviceCount);
    devices = (cl_device_id*)malloc(sizeof(cl_device_id) * deviceCount);
    clGetDeviceIDs(pid, CL_DEVICE_TYPE_ALL, deviceCount, devices, NULL);

    // for each device print critical attributes
    for (int j = 0; j < deviceCount; j++) {

        // print device name
        clGetDeviceInfo(devices[j], CL_DEVICE_NAME, 0, NULL, &valueSize);
        value = (char*)malloc(valueSize);
        clGetDeviceInfo(devices[j], CL_DEVICE_NAME, valueSize, value, NULL);
        printf("%d. Device: %s\n", j + 1, value);
        free(value);

        // print hardware device version
        clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, 0, NULL, &valueSize);
        value = (char*)malloc(valueSize);
        clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, valueSize, value, NULL);
        printf(" %d.%d Hardware version: %s\n", j + 1, 1, value);
        free(value);

        // print software driver version
        clGetDeviceInfo(devices[j], CL_DRIVER_VERSION, 0, NULL, &valueSize);
        value = (char*)malloc(valueSize);
        clGetDeviceInfo(devices[j], CL_DRIVER_VERSION, valueSize, value, NULL);
        printf(" %d.%d Software version: %s\n", j + 1, 2, value);
        free(value);

        // print c version supported by compiler for device
        clGetDeviceInfo(devices[j], CL_DEVICE_OPENCL_C_VERSION, 0, NULL, &valueSize);
        value = (char*)malloc(valueSize);
        clGetDeviceInfo(devices[j], CL_DEVICE_OPENCL_C_VERSION, valueSize, value, NULL);
        printf(" %d.%d OpenCL C version: %s\n", j + 1, 3, value);
        free(value);

        // print parallel compute units
        clGetDeviceInfo(devices[j], CL_DEVICE_MAX_COMPUTE_UNITS,
            sizeof(maxComputeUnits), &maxComputeUnits, NULL);
        printf(" %d.%d Parallel compute units: %d\n", j + 1, 4, maxComputeUnits);

    }

}


void mGenerateMatrices2D(float inputArray[][dim], int height, int width) {
    cl_float temp = 0;

    std::random_device rd;
    srand((unsigned int)rd());
    cl_float interval = 2.0;
    cl_float lowerLimit = -1.0;

    for (cl_uint i = 0; i < height; ++i){
        for (cl_uint j = 0; j < width; j++) {
            temp = lowerLimit + (cl_float(rand()) / cl_float((RAND_MAX)) * interval);
            inputArray[i][j] = temp;
        }
    }
}

/*void matrixmultestCPUbasic() {
    int mDim = dim, pDim = dim, nDim = dim;
    /*cl_uint optimizedSizeTempA = ((sizeof(cl_float) * mDim * pDim - 1) / 64 + 1) * 64;
    cl_uint optimizedSizeTempB = ((sizeof(cl_float) * pDim * nDim - 1) / 64 + 1) * 64;
    cl_uint optimizedSizeTempC = ((sizeof(cl_float) * mDim * nDim - 1) / 64 + 1) * 64;
    cl_float* matrixAar = (cl_float*)_aligned_malloc(optimizedSizeTempA, 4096);
    cl_float* matrixBar = (cl_float*)_aligned_malloc(optimizedSizeTempB, 4096);
    cl_float* matrixCar = (cl_float*)_aligned_malloc(optimizedSizeTempC, 4096);
    mGenerateMatrices(matrixAar, mDim, pDim);
    mGenerateMatrices(matrixBar, pDim, nDim);*/
    //mGenerateMatrices(matrixCar, mDim, nDim);

    /*float matrixAar[dim][dim];
    float matrixBar[dim][dim];
    float matrixCar[dim][dim];
    mGenerateMatrices2D(matrixAar, dim, dim);
    mGenerateMatrices2D(matrixBar, dim, dim);

    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            matrixCar[i][j] = 0.0;
        }
    }

    //MAYBE ADD CODE TO MAKE SURE matrixCar starts with 0 values

    using namespace std::chrono;

    auto start = high_resolution_clock::now();
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            for (int k = 0; k < dim; k++) {
                matrixCar[i][j] += (matrixAar[i][k]) * (matrixBar[k][j]); //MAYBE IT'S SLOW DUE TO MULTIPLICATIONS
            }
        }
    }
    auto elapsed = std::chrono::high_resolution_clock::now() - start;
    long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    std::cout << microseconds << " microseconds since epoch\n";
    /*for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            std::cout << matrixAar[i][j] << " ";
        }
        std::cout << '\n';
    }
    std::cout << '\n';
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            std::cout << matrixBar[i][j] << " ";
        }
        std::cout << '\n';
    }
    std::cout << '\n';
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            std::cout << matrixCar[i][j] << " ";
        }
        std::cout << '\n';
    }
    return;
}*/

cl_uint initializeparamsClassifier(ocl_args_d_t* ocl, cl_mem** buffersWeightsArray, cl_mem** buffersOutsArray, cl_mem** buffersDeltasArray, cl_mem* bufferInputArray, cl_float** costs,
    int dimensions[], int batchSize, int iterations, int layers, cl_float** correctOutput, uchar** dataset, uchar* labels) {

    cl_int err = CL_SUCCESS;
    cl_uint optimizedSize = ((sizeof(cl_mem) * layers - 1) / 64 + 1) * 64;
    *buffersWeightsArray = (cl_mem*)_aligned_malloc(optimizedSize, 4096); //array of memory objects, where each memory object is a buffer of weights between layers

    *buffersOutsArray = (cl_mem*)_aligned_malloc(optimizedSize, 4096); //array of memory objects, where each memory object is a buffer image of outputs of layers

    *buffersDeltasArray = (cl_mem*)_aligned_malloc(optimizedSize, 4096); //array of memory objects, where each memory object is an image of deltas of layers

    cl_uint optimizedSizeIn = ((sizeof(cl_float) * dimensions[0] * batchSize - 1) / 64 + 1) * 64;
    cl_float* inArray = (cl_float*)_aligned_malloc(optimizedSizeIn, 4096); //array of network input
    for (int i = 0; i < batchSize; ++i) {
        for (int j = 0; j < dimensions[0]; ++j) {
            inArray[j * batchSize + i] = (cl_float)dataset[i][j];
        }
    }
    cl_uint optimizedSizeOut = ((sizeof(cl_float) * batchSize - 1) / 64 + 1) * 64;
    *correctOutput = (cl_float*)_aligned_malloc(optimizedSizeOut, 4096); //array of network input
    for (int i = 0; i < batchSize; ++i) {
        (*correctOutput)[i] = (cl_float)labels[i];
    }


    if (NULL == *buffersWeightsArray || NULL == *buffersOutsArray || NULL == *buffersDeltasArray || NULL == inArray || NULL == *correctOutput)
    {
        LogError("Error: _aligned_malloc failed to allocate buffers.\n");
        return -1;
    }

    *bufferInputArray = clCreateBuffer(ocl->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * dimensions[0] * batchSize, inArray, &err);
    if (CL_SUCCESS != err)
    {
        LogError("Error: creating input buffer returned %s\n", TranslateOpenCLError(err));
        return err;
    }

    int optimizedSizeCosts = ((sizeof(cl_float) * iterations - 1) / 64 + 1) * 64;
    *costs = (cl_float*)_aligned_malloc(optimizedSizeCosts, 4096);

    cl_uint optimizedSize1;
    cl_uint optimizedSize2;
    cl_float* tempWeightArray;
    int mDim, pDim;
    for (cl_uint x = 0; x < layers; ++x) {
        mDim = dimensions[x + 1];
        pDim = dimensions[x];
        optimizedSize1 = ((sizeof(cl_float) * mDim * pDim - 1) / 64 + 1) * 64;
        tempWeightArray = (cl_float*)_aligned_malloc(optimizedSize1, 4096);
        std::cout << "Weights of layer " << x << " are: \n";
        mGenerateMatrices(tempWeightArray, mDim, pDim);


        // Create first buffer based on host memory inputA
        (*buffersWeightsArray)[x] = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * mDim * pDim, tempWeightArray, &err);
        if (CL_SUCCESS != err)
        {
            LogError("Error: clCreateImage for srcA returned %s\n", TranslateOpenCLError(err));
            return err;
        }

        (*buffersOutsArray)[x] = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE, sizeof(cl_float) * mDim * batchSize, NULL, &err);
        if (CL_SUCCESS != err)
        {
            LogError("Error: clCreateImage for srcA returned %s\n", TranslateOpenCLError(err));
            return err;
        }

        (*buffersDeltasArray)[x] = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE, sizeof(cl_float) * mDim * batchSize, NULL, &err);
        if (CL_SUCCESS != err)
        {
            LogError("Error: clCreateImage for srcA returned %s\n", TranslateOpenCLError(err));
            return err;
        }
        _aligned_free(tempWeightArray);
    }
}

cl_uint forwardpassClassifier(ocl_args_d_t* ocl, cl_mem* buffersWeightsArray, cl_mem* buffersOutsArray, cl_mem* bufferInputArray, 
    int dimensions[], int* ActivationFunctions, cl_kernel* activationFunctionKernels, cl_kernel* activationFunctionKernelsSimple, int batchSize, int layers) {

    std::cout << "In forwardprop \n";
    ocl->dstMem = *bufferInputArray;
    int mDim, pDim, nDim = batchSize;
    for (int x = 0; x < layers; ++x) {

        mDim = dimensions[x + 1];
        pDim = dimensions[x];

        ocl->srcA = buffersWeightsArray[x];
        ocl->srcB = ocl->dstMem;
        ocl->dstMem = buffersOutsArray[x];

        // Program consists of kernels.
        // Each kernel can be called (enqueued) from the host part of OpenCL application.
        // To call the kernel, you need to create it from existing program.
        if (mDim % 16 == 0 && nDim % 16 == 0) {
            ocl->kernel = activationFunctionKernels[ActivationFunctions[x]];

        }
        else {
            ocl->kernel = activationFunctionKernelsSimple[ActivationFunctions[x]];
        }

        // Passing arguments into OpenCL kernel.
        if (CL_SUCCESS != mSetKernelArguments(ocl, NULL, mDim, pDim, nDim, 0.0, 1))
        {
            return -1;
        }
        //system("pause");
        if (CL_SUCCESS != mExecuteMultiplyKernel(ocl, mDim, nDim))
        {
            return -1;
        }
        //system("pause");
    }
}



cl_uint backpropClassifier(ocl_args_d_t* ocl, cl_mem* buffersWeightsArray, cl_mem* buffersOutsArray, cl_mem* buffersDeltasArray, cl_mem* bufferInputArray, int dimensions[],
    int* ActivationFunctions, cl_kernel* activationFunctionDeltaKernels, cl_kernel* activationFunctionDeltaKernelsSimple, cl_float* correctOutput, cl_float* costs, int iter, int batchSize, int layers, int classes) {

    std::cout << "In backprop for iter " << iter << " \n";
    cl_int err = CL_SUCCESS;

    cl_uint optimizedSizeNetworkOutput = ((sizeof(cl_float) * batchSize*classes - 1) / 64 + 1) * 64;
    cl_float* preSoftmaxOutputs = (cl_float*)_aligned_malloc(optimizedSizeNetworkOutput, 4096);
    cl_float* softmaxOutputs = (cl_float*)_aligned_malloc(optimizedSizeNetworkOutput, 4096);
    cl_float* deltas = (cl_float*)_aligned_malloc(optimizedSizeNetworkOutput, 4096);

    cl_uint optimizedSizeNetworkOutput2 = ((sizeof(int) * batchSize - 1) / 64 + 1) * 64;
    int* choices = (int*)_aligned_malloc(optimizedSizeNetworkOutput2, 4096);

    err = clEnqueueReadBuffer(ocl->commandQueue, ocl->dstMem, true, 0, sizeof(cl_float) * classes * batchSize, preSoftmaxOutputs, 0, NULL, NULL);
    if (CL_SUCCESS != err)
    {
        LogError("Error: clEnqueueReadBuffer returned %s\n", TranslateOpenCLError(err));
    }

    int idx, correctClass;
    cl_float maxval, temp;
    for (int i = 0; i < batchSize; ++i) {
        //std::cout << "Entering calculation for batch element " << i << "\n";
        idx = 0;
        temp = 0.0f;
        maxval = -FLT_MAX;
        correctClass = (int)correctOutput[i];
        for (int j = 0; j < classes; ++j) {
            //std::cout << "presoftmax for " << j << " th class and batch item " << i << " is " << preSoftmaxOutputs[j * batchSize + i] << '\n';
            if (preSoftmaxOutputs[j * batchSize + i] > maxval) {
                idx = j;
                maxval = preSoftmaxOutputs[j * batchSize + i];
            }
        }
        if (iter % 20 == 0 && iter>0) {
            std::cout << "Predicted class was " << idx << '\n';
            std::cout << "Correct class was " << correctClass << '\n';
        }
        choices[i] = idx;
        for (int j = 0; j < classes; ++j) {
            preSoftmaxOutputs[j * batchSize + i] = preSoftmaxOutputs[j * batchSize + i] - maxval;
            //std::cout << "presoft recentered for class "<<j<<" is " << preSoftmaxOutputs[j * batchSize + i] << "\n";
            softmaxOutputs[j * batchSize + i] = exp(preSoftmaxOutputs[j * batchSize + i]);
            temp += exp(preSoftmaxOutputs[j * batchSize + i]);
        }
        for (int j = 0; j < classes; ++j) {
            softmaxOutputs[j * batchSize + i] = softmaxOutputs[j * batchSize + i] /temp;
        }
    }

    //These delta calculation formulas correspond to a CEL cost function
    for (int i = 0; i < batchSize; ++i) {
        correctClass = (int)correctOutput[i];
        for (int j = 0; j < classes; ++j) {
            if (j == correctClass) {
                deltas[j * batchSize + i] = softmaxOutputs[j * batchSize + i] - 1.0f;
            }
            else {
                deltas[j * batchSize + i] = softmaxOutputs[j * batchSize + i];
            }
            //std::cout<<"Delta for batch item "<<i<<" and class "<<j<<" is "<< deltas[j * batchSize + i]<<'\n';
        }
    }

    //Here I am actually returning the accuracy
    costs[iter] = AccuracyFunction(correctOutput, choices, batchSize);

    err = clEnqueueWriteBuffer(ocl->commandQueue, buffersDeltasArray[layers - 1], true, 0, sizeof(cl_float) * batchSize*classes, deltas, 0, NULL, NULL);
    if (CL_SUCCESS != err)
    {
        LogError("Error: clEnqueueReadBuffer returned %s\n", TranslateOpenCLError(err));
    }

    //non-output deltas calculation loop
    ocl->dstMem = buffersDeltasArray[layers - 1];
    cl_mem outputs;
    int mDim, pDim, nDim = batchSize;
    for (int x = layers - 1; x > 0; --x) {

        mDim = dimensions[x];
        pDim = dimensions[x + 1];
        ocl->srcA = buffersWeightsArray[x];
        ocl->srcB = ocl->dstMem;
        ocl->dstMem = buffersDeltasArray[x - 1];
        outputs = buffersOutsArray[x - 1];

        if (mDim % 16 == 0 && nDim % 16 == 0) {
            ocl->kernel = activationFunctionDeltaKernels[ActivationFunctions[x]];

        }
        else {
            ocl->kernel = activationFunctionDeltaKernelsSimple[ActivationFunctions[x]];
        }

        if (CL_SUCCESS != mSetKernelArguments(ocl, &outputs, mDim, pDim, nDim, 0.0, 2))
        {
            return -1;
        }

        if (CL_SUCCESS != mExecuteMultiplyKernel(ocl, mDim, nDim)) //thankfully, the same function works for our Multiply_Deltas kernel too
        {
            return -1;
        }
    }

    //perform weight updates now. We can potentially parallelize this fully even across all network layers
    //but for now it's only across the weights of each layer and then sequentially across layers
    cl_float learning_rate = 0.0008;
    pDim = batchSize;
    for (int x = layers - 1; x >= 0; --x) {
        //std::cout << "I'm in iteration " << x << " of the weight update loop \n";
        mDim = dimensions[x + 1];
        nDim = dimensions[x];
        ocl->srcA = buffersDeltasArray[x];
        ocl->dstMem = buffersWeightsArray[x];
        if (x != 0) {
            ocl->srcB = buffersOutsArray[x - 1];
        }
        else {
            ocl->srcB = *bufferInputArray;
        }

        if (mDim % 16 == 0 && nDim % 16 == 0) {
            ocl->kernel = clCreateKernel(ocl->program, "Update_Weights_Buffers", &err);
            if (CL_SUCCESS != err)
            {
                LogError("Error: clCreateKernel returned %s\n", TranslateOpenCLError(err));
                return -1;
            }
        }
        else {
            ocl->kernel = clCreateKernel(ocl->programSimple, "Update_Weights_Buffers", &err);
            if (CL_SUCCESS != err)
            {
                LogError("Error: clCreateKernel returned %s\n", TranslateOpenCLError(err));
                return -1;
            }
        }

        if (CL_SUCCESS != mSetKernelArguments(ocl, NULL, mDim, pDim, nDim, learning_rate, 3))
        {
            return -1;
        }

        if (CL_SUCCESS != mExecuteMultiplyKernel(ocl, mDim, nDim)) //thankfully, the same function works for our Multiply_Deltas kernel too
        {
            return -1;
        }
    }
}

cl_uint validationClassifier(ocl_args_d_t* ocl, cl_mem* buffersWeightsArray, cl_mem* buffersOutsArray, cl_mem* bufferInputArray, int dimensions[],
    int* ActivationFunctions, cl_kernel* activationFunctionKernels, cl_kernel* activationFunctionKernelsSimple, int layers, int classes, uchar** valDataset, uchar* valLabels, int numValImages ) {
    
    std::cout << "In validation \n";

    cl_int err = CL_SUCCESS;

    cl_uint optimizedSizeIn = ((sizeof(cl_float) * dimensions[0] * numValImages - 1) / 64 + 1) * 64;
    cl_float* inArray = (cl_float*)_aligned_malloc(optimizedSizeIn, 4096); //array of network input

    for (int i = 0; i < numValImages; ++i) {
        for (int j = 0; j < dimensions[0]; ++j) {
            inArray[j * numValImages + i] = (cl_float)valDataset[i][j];
        }
    }
    cl_uint optimizedSizeOut = ((sizeof(cl_float) * classes * numValImages - 1) / 64 + 1) * 64;
    cl_float* correctOutput = (cl_float*)_aligned_malloc(optimizedSizeOut, 4096); //array of network ground truth

    for (int i = 0; i < numValImages; ++i) {
        correctOutput[i] = (cl_float)valLabels[i];
    }


    if (NULL == inArray || NULL == correctOutput)
    {
        LogError("Error: _aligned_malloc failed to allocate buffers.\n");
        return -1;
    }

    *bufferInputArray = clCreateBuffer(ocl->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * dimensions[0] * numValImages, inArray, &err);
    if (CL_SUCCESS != err)
    {
        LogError("Error: creating input buffer returned %s\n", TranslateOpenCLError(err));
        return err;
    }


    int mDim, pDim, nDim = numValImages;
    for (cl_uint x = 0; x < layers; ++x) {
        mDim = dimensions[x + 1];
        pDim = dimensions[x];

        (buffersOutsArray)[x] = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE, sizeof(cl_float) * mDim * nDim, NULL, &err);
        if (CL_SUCCESS != err)
        {
            LogError("Error: clCreateImage for srcA returned %s\n", TranslateOpenCLError(err));
            return err;
        }
    }

    ocl->dstMem = *bufferInputArray;
    for (int x = 0; x < layers; ++x) {

        mDim = dimensions[x + 1];
        pDim = dimensions[x];

        ocl->srcA = buffersWeightsArray[x];
        ocl->srcB = ocl->dstMem;
        ocl->dstMem = buffersOutsArray[x];

        // Program consists of kernels.
        // Each kernel can be called (enqueued) from the host part of OpenCL application.
        // To call the kernel, you need to create it from existing program.
        if (mDim % 16 == 0 && nDim % 16 == 0) {
            ocl->kernel = activationFunctionKernels[ActivationFunctions[x]];

        }
        else {
            ocl->kernel = activationFunctionKernelsSimple[ActivationFunctions[x]];
        }

        // Passing arguments into OpenCL kernel.
        if (CL_SUCCESS != mSetKernelArguments(ocl, NULL, mDim, pDim, nDim, 0.0, 1))
        {
            return -1;
        }
        if (CL_SUCCESS != mExecuteMultiplyKernel(ocl, mDim, nDim))
        {
            return -1;
        }
    }

    cl_uint optimizedSizeNetworkOutput = ((sizeof(cl_float) * numValImages * classes - 1) / 64 + 1) * 64;
    cl_float* preSoftmaxOutputs = (cl_float*)_aligned_malloc(optimizedSizeNetworkOutput, 4096);
    cl_float* softmaxOutputs = (cl_float*)_aligned_malloc(optimizedSizeNetworkOutput, 4096);

    cl_uint optimizedSizeNetworkOutput2 = ((sizeof(int) * numValImages - 1) / 64 + 1) * 64;
    int* choices = (int*)_aligned_malloc(optimizedSizeNetworkOutput2, 4096);

    err = clEnqueueReadBuffer(ocl->commandQueue, ocl->dstMem, true, 0, sizeof(cl_float) * classes * numValImages, preSoftmaxOutputs, 0, NULL, NULL);
    if (CL_SUCCESS != err)
    {
        LogError("Error: clEnqueueReadBuffer returned %s\n", TranslateOpenCLError(err));
    }

    int idx, correctClass;
    cl_float maxval, temp;
    for (int i = 0; i < numValImages; ++i) {
        //std::cout << "Entering calculation for batch element " << i << "\n";
        idx = 0;
        temp = 0.0f;
        maxval = -FLT_MAX;
        correctClass = (int)correctOutput[i];
        for (int j = 0; j < classes; ++j) {
            //std::cout << "presoftmax for " << j << " th class and batch item " << i << " is " << preSoftmaxOutputs[j * batchSize + i] << '\n';
            if (preSoftmaxOutputs[j * numValImages + i] > maxval) {
                idx = j;
                maxval = preSoftmaxOutputs[j * numValImages + i];
            }
        }
        choices[i] = idx;
        for (int j = 0; j < classes; ++j) {
            preSoftmaxOutputs[j * numValImages + i] = preSoftmaxOutputs[j * numValImages + i] - maxval;
            //std::cout << "presoft recentered for class "<<j<<" is " << preSoftmaxOutputs[j * batchSize + i] << "\n";
            softmaxOutputs[j * numValImages + i] = exp(preSoftmaxOutputs[j * numValImages + i]);
            temp += exp(preSoftmaxOutputs[j * numValImages + i]);
        }
        for (int j = 0; j < classes; ++j) {
            softmaxOutputs[j * numValImages + i] = softmaxOutputs[j * numValImages + i] / temp;
        }
    }

    //Here I am actually returning the accuracy
    cl_float accuracy = AccuracyFunction(correctOutput, choices, numValImages);
    std::cout << "Validation set accuracy is " << accuracy << '\n';
}

cl_uint initializeparams(ocl_args_d_t* ocl, cl_mem** buffersWeightsArray, cl_mem** buffersOutsArray, cl_mem** buffersDeltasArray, cl_mem* bufferInputArray, cl_float** costs,
    int dimensions[], int batchSize, int iterations, int layers) {

    cl_int err = CL_SUCCESS;
    cl_uint optimizedSize = ((sizeof(cl_mem) * layers - 1) / 64 + 1) * 64;
    *buffersWeightsArray = (cl_mem*)_aligned_malloc(optimizedSize, 4096); //array of memory objects, where each memory object is a buffer of weights between layers

    *buffersOutsArray = (cl_mem*)_aligned_malloc(optimizedSize, 4096); //array of memory objects, where each memory object is a buffer image of outputs of layers

    *buffersDeltasArray = (cl_mem*)_aligned_malloc(optimizedSize, 4096); //array of memory objects, where each memory object is an image of deltas of layers

    cl_uint optimizedSizeIn = ((sizeof(cl_float) * dimensions[0] * batchSize - 1) / 64 + 1) * 64;
    cl_float* inArray = (cl_float*)_aligned_malloc(optimizedSizeIn, 4096); //array of network input
    std::cout << "Matrix input is \n";
    mGenerateMatrices(inArray, dimensions[0], batchSize);//TODO: have some code here to initiliaze inputArray with external training data
    if (NULL == *buffersWeightsArray || NULL == *buffersOutsArray || NULL == *buffersDeltasArray || NULL == inArray)
    {
        LogError("Error: _aligned_malloc failed to allocate buffers.\n");
        return -1;
    }

    *bufferInputArray = clCreateBuffer(ocl->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * dimensions[0] * batchSize, inArray, &err);
    if (CL_SUCCESS != err)
    {
        LogError("Error: creating input buffer returned %s\n", TranslateOpenCLError(err));
        return err;
    }

    int optimizedSizeCosts = ((sizeof(cl_float) * iterations - 1) / 64 + 1) * 64;
    *costs = (cl_float*)_aligned_malloc(optimizedSizeCosts, 4096);

    cl_uint optimizedSize1;
    cl_uint optimizedSize2;
    cl_float* tempWeightArray;
    int mDim, pDim;
    for (cl_uint x = 0; x < layers; ++x) {
        mDim = dimensions[x + 1];
        pDim = dimensions[x];
        optimizedSize1 = ((sizeof(cl_float) * mDim * pDim - 1) / 64 + 1) * 64;
        tempWeightArray = (cl_float*)_aligned_malloc(optimizedSize1, 4096);
        std::cout << "Weights of layer " << x << " are: \n";
        mGenerateMatrices(tempWeightArray, mDim, pDim);


        // Create first buffer based on host memory inputA
        (*buffersWeightsArray)[x] = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * mDim * pDim, tempWeightArray, &err);
        if (CL_SUCCESS != err)
        {
            LogError("Error: clCreateImage for srcA returned %s\n", TranslateOpenCLError(err));
            return err;
        }

        (*buffersOutsArray)[x] = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE, sizeof(cl_float) * mDim * batchSize, NULL, &err);
        if (CL_SUCCESS != err)
        {
            LogError("Error: clCreateImage for srcA returned %s\n", TranslateOpenCLError(err));
            return err;
        }

        (*buffersDeltasArray)[x] = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE, sizeof(cl_float) * mDim * batchSize, NULL, &err);
        if (CL_SUCCESS != err)
        {
            LogError("Error: clCreateImage for srcA returned %s\n", TranslateOpenCLError(err));
            return err;
        }
        _aligned_free(tempWeightArray);
    }
}

cl_uint forwardpass(ocl_args_d_t* ocl, cl_mem* buffersWeightsArray, cl_mem* buffersOutsArray, cl_mem* bufferInputArray, int dimensions[], int* ActivationFunctions, cl_kernel* activationFunctionKernels, int batchSize, int layers) {

    std::cout << "In forwardprop \n";
    ocl->dstMem = *bufferInputArray;
    int mDim, pDim, nDim = batchSize;
    for (int x = 0; x < layers; ++x) {

        mDim = dimensions[x + 1];
        pDim = dimensions[x];

        ocl->srcA = buffersWeightsArray[x];
        ocl->srcB = ocl->dstMem;
        ocl->dstMem = buffersOutsArray[x];

        // Program consists of kernels.
        // Each kernel can be called (enqueued) from the host part of OpenCL application.
        // To call the kernel, you need to create it from existing program.
        ocl->kernel = activationFunctionKernels[ActivationFunctions[x]];

        // Passing arguments into OpenCL kernel.
        if (CL_SUCCESS != mSetKernelArguments(ocl, NULL, mDim, pDim, nDim, 0.0, 1))
        {
            return -1;
        }
        //system("pause");
        if (CL_SUCCESS != mExecuteMultiplyKernel(ocl, mDim, nDim))
        {
            return -1;
        }
        //system("pause");
    }
}


cl_uint backprop(ocl_args_d_t* ocl, cl_mem* buffersWeightsArray, cl_mem* buffersOutsArray, cl_mem* buffersDeltasArray, cl_mem* bufferInputArray, int dimensions[],
    int* ActivationFunctions, cl_kernel* activationFunctionDeltaKernels, cl_float* correctOutput, cl_float* costs, int iter, int batchSize, int layers) {

    std::cout << "In backprop for iter " << iter << " \n";
    cl_int err = CL_SUCCESS;

    cl_uint optimizedSizeNetworkOutput = ((sizeof(cl_float) * batchSize - 1) / 64 + 1) * 64;
    cl_float* resultPtr1 = (cl_float*)_aligned_malloc(optimizedSizeNetworkOutput, 4096);
    cl_float* networkOutput = (cl_float*)_aligned_malloc(optimizedSizeNetworkOutput, 4096);
    err = clEnqueueReadBuffer(ocl->commandQueue, ocl->dstMem, true, 0, sizeof(cl_float) * batchSize, resultPtr1, 0, NULL, NULL);
    if (CL_SUCCESS != err)
    {
        LogError("Error: clEnqueueReadBuffer returned %s\n", TranslateOpenCLError(err));
    }
    for (int x = 0; x < batchSize; ++x) {
        networkOutput[x] = resultPtr1[x];
    }

    costs[iter] = MSECostFunction(correctOutput, resultPtr1, batchSize);

    //Then, let's compute the deltas, starting with the output delta 
    for (int x = 0; x < batchSize; ++x) {
        resultPtr1[x] = resultPtr1[x] - correctOutput[x];
    }

    if (ActivationFunctions[layers - 1] == 0) {
        for (int x = 0; x < batchSize; ++x) {
            networkOutput[x] = 1.0;
        }
    }
    else if (ActivationFunctions[layers - 1] == 1) {
        for (int x = 0; x < batchSize; ++x) {
            networkOutput[x] = networkOutput[x] * (1.0 - networkOutput[x]);
        }
    }
    else if (ActivationFunctions[layers - 1] == 2) {
        for (int x = 0; x < batchSize; ++x) {
            networkOutput[x] = 1.0 - networkOutput[x] * networkOutput[x];
        }
    }
    else if (ActivationFunctions[layers - 1] == 3) {
        for (int x = 0; x < batchSize; ++x) {
            if (networkOutput[x] == 0.0)
                networkOutput[x] = 0.0;
            else
                networkOutput[x] = 1.0;
        }
    }

    for (int x = 0; x < batchSize; ++x) {
        resultPtr1[x] = resultPtr1[x] * networkOutput[x];
    }
    err = clEnqueueWriteBuffer(ocl->commandQueue, buffersDeltasArray[layers - 1], true, 0, sizeof(cl_float) * batchSize, resultPtr1, 0, NULL, NULL);
    if (CL_SUCCESS != err)
    {
        LogError("Error: clEnqueueReadBuffer returned %s\n", TranslateOpenCLError(err));
    }
    _aligned_free(resultPtr1);
    _aligned_free(networkOutput);

    //non-output deltas calculation loop
    ocl->dstMem = buffersDeltasArray[layers - 1];
    cl_mem outputs;
    int mDim, pDim, nDim = batchSize;
    for (int x = layers - 1; x > 0; --x) {

        mDim = dimensions[x];
        pDim = dimensions[x + 1];
        ocl->srcA = buffersWeightsArray[x];
        ocl->srcB = ocl->dstMem;
        ocl->dstMem = buffersDeltasArray[x - 1];
        outputs = buffersOutsArray[x - 1];

        // Program consists of kernels.
        // Each kernel can be called (enqueued) from the host part of OpenCL application.
        // To call the kernel, you need to create it from existing program.
        ocl->kernel = activationFunctionDeltaKernels[ActivationFunctions[x]];
        // Passing arguments into OpenCL kernel.
        if (CL_SUCCESS != mSetKernelArguments(ocl, &outputs, mDim, pDim, nDim, 0.0, 2))
        {
            return -1;
        }

        // Execute (enqueue) the kernel
        if (CL_SUCCESS != mExecuteMultiplyKernel(ocl, mDim, nDim)) //thankfully, the same function works for our Multiply_Deltas kernel too
        {
            return -1;
        }
    }

    //perform weight updates now. We can potentially parallelize this fully even across all network layers
    //but for now it's only across the weights of each layer and then sequentially across layers
    cl_float learning_rate = 0.001;
    for (int x = layers - 1; x >= 0; --x) {
        //std::cout << "I'm in iteration " << x << " of the weight update loop \n";
        mDim = dimensions[x + 1];
        nDim = dimensions[x];
        ocl->srcA = buffersDeltasArray[x];
        ocl->dstMem = buffersWeightsArray[x];
        if (x != 0) {
            ocl->srcB = buffersOutsArray[x - 1];
        }
        else {
            ocl->srcB = *bufferInputArray;
        }

        ocl->kernel = clCreateKernel(ocl->program, "Update_Weights_Buffers", &err);
        if (CL_SUCCESS != err)
        {
            LogError("Error: clCreateKernel returned %s\n", TranslateOpenCLError(err));
            return -1;
        }

        if (CL_SUCCESS != mSetKernelArguments(ocl, NULL, mDim, nDim, batchSize, learning_rate, 3))
        {
            return -1;
        }

        if (CL_SUCCESS != mExecuteMultiplyKernel(ocl, mDim, nDim)) //thankfully, the same function works for our Multiply_Deltas kernel too
        {
            return -1;
        }
    }
}

cl_uint forwardpass1(ocl_args_d_t *ocl, cl_mem *buffersWeightsArray, cl_mem *buffersOutsArray, cl_mem *bufferInputArray, int dimensions[],int* ActivationFunctions, cl_kernel *activationFunctionKernels, int batchSize, int layers) {
        
        std::cout << "In forwardprop \n";
        ocl->dstMem = *bufferInputArray;
        int mDim, pDim, nDim = batchSize;
        for (int x = 0; x < layers; ++x) {

            mDim = dimensions[x + 1];
            pDim = dimensions[x];

            ocl->srcA = buffersWeightsArray[x];
            ocl->srcB = ocl->dstMem;
            ocl->dstMem = buffersOutsArray[x];

            // Program consists of kernels.
            // Each kernel can be called (enqueued) from the host part of OpenCL application.
            // To call the kernel, you need to create it from existing program.
            ocl->kernel = activationFunctionKernels[ActivationFunctions[x]];

            // Passing arguments into OpenCL kernel.
            if (CL_SUCCESS != mSetKernelArguments(ocl, NULL, mDim, pDim, nDim, 0.0, 1))
            {
                return -1;
            }
            //system("pause");
            if (CL_SUCCESS != mExecuteMultiplyKernel(ocl, mDim, nDim))
            {
                return -1;
            }
            //system("pause");
        }
}

cl_uint backprop1(ocl_args_d_t* ocl, cl_mem* buffersWeightsArray, cl_mem* buffersOutsArray, cl_mem* buffersDeltasArray, cl_mem* bufferInputArray, int dimensions[], 
    int* ActivationFunctions, cl_kernel* activationFunctionDeltaKernels, cl_float *correctOutput, cl_float *costs, int iter, int batchSize, int layers) {
    
    std::cout << "In backprop for iter " << iter << " \n";
    cl_int err = CL_SUCCESS;

    cl_uint optimizedSizeNetworkOutput = ((sizeof(cl_float) * batchSize - 1) / 64 + 1) * 64;
    cl_float* resultPtr1 = (cl_float*)_aligned_malloc(optimizedSizeNetworkOutput, 4096);
    cl_float* networkOutput = (cl_float*)_aligned_malloc(optimizedSizeNetworkOutput, 4096);
    err = clEnqueueReadBuffer(ocl->commandQueue, ocl->dstMem, true, 0, sizeof(cl_float) * batchSize, resultPtr1, 0, NULL, NULL);
    if (CL_SUCCESS != err)
    {
        LogError("Error: clEnqueueReadBuffer returned %s\n", TranslateOpenCLError(err));
    }
    for (int x = 0; x < batchSize; ++x) {
        networkOutput[x] = resultPtr1[x];
    }

    costs[iter] = MSECostFunction(correctOutput, resultPtr1, batchSize);

    //Then, let's compute the deltas, starting with the output delta 
    for (int x = 0; x < batchSize; ++x) {
        resultPtr1[x] = resultPtr1[x] - correctOutput[x];
    }

    if (ActivationFunctions[layers - 1] == 0) {
        for (int x = 0; x < batchSize; ++x) {
            networkOutput[x] = 1.0;
        }
    }
    else if (ActivationFunctions[layers - 1] == 1) {
        for (int x = 0; x < batchSize; ++x) {
            networkOutput[x] = networkOutput[x] * (1.0 - networkOutput[x]);
        }
    }
    else if (ActivationFunctions[layers - 1] == 2) {
        for (int x = 0; x < batchSize; ++x) {
            networkOutput[x] = 1.0 - networkOutput[x] * networkOutput[x];
        }
    }
    else if (ActivationFunctions[layers - 1] == 3) {
        for (int x = 0; x < batchSize; ++x) {
            if (networkOutput[x] == 0.0)
                networkOutput[x] = 0.0;
            else
                networkOutput[x] = 1.0;
        }
    }
    for (int x = 0; x < batchSize; ++x) {
        // std::cout << "Output delta function term "<<x<<" is: " << networkOutput[x] << '\n';
        // std::cout << "Output delta pre function term " << x << "is: " << resultPtr1[x] << '\n';
    }

    for (int x = 0; x < batchSize; ++x) {
        resultPtr1[x] = resultPtr1[x] * networkOutput[x];
    }
    for (int x = 0; x < batchSize; ++x) {
        // std::cout << "Output delta term "<<x<<" is: " << resultPtr1[x] << '\n';
    }
    err = clEnqueueWriteBuffer(ocl->commandQueue, buffersDeltasArray[layers - 1], true, 0, sizeof(cl_float) * batchSize, resultPtr1, 0, NULL, NULL);
    if (CL_SUCCESS != err)
    {
        LogError("Error: clEnqueueReadBuffer returned %s\n", TranslateOpenCLError(err));
    }
    _aligned_free(resultPtr1);
    _aligned_free(networkOutput);

    //non-output deltas calculation loop
    ocl->dstMem = buffersDeltasArray[layers - 1];
    cl_mem outputs;
    int mDim, pDim, nDim = batchSize;
    for (int x = layers - 1; x > 0; --x) {

        mDim = dimensions[x];
        pDim = dimensions[x + 1];
        ocl->srcA = buffersWeightsArray[x];
        ocl->srcB = ocl->dstMem;
        ocl->dstMem = buffersDeltasArray[x - 1];
        outputs = buffersOutsArray[x - 1];

        // Program consists of kernels.
        // Each kernel can be called (enqueued) from the host part of OpenCL application.
        // To call the kernel, you need to create it from existing program.
        ocl->kernel = activationFunctionDeltaKernels[ActivationFunctions[x]];
        // Passing arguments into OpenCL kernel.
        if (CL_SUCCESS != mSetKernelArguments(ocl, &outputs, mDim, pDim, nDim, 0.0, 2))
        {
            return -1;
        }

        // Execute (enqueue) the kernel
        if (CL_SUCCESS != mExecuteMultiplyKernel(ocl, mDim, nDim)) //thankfully, the same function works for our Multiply_Deltas kernel too
        {
            return -1;
        }
    }

    //perform weight updates now. We can potentially parallelize this fully even across all network layers
    //but for now it's only across the weights of each layer and then sequentially across layers
    cl_float learning_rate = 0.06;
    for (int x = layers - 1; x >= 0; --x) {
        //std::cout << "I'm in iteration " << x << " of the weight update loop \n";
        mDim = dimensions[x + 1];
        nDim = dimensions[x];
        ocl->srcA = buffersDeltasArray[x];
        ocl->dstMem = buffersWeightsArray[x];
        if (x != 0) {
            ocl->srcB = buffersOutsArray[x - 1];
        }
        else {
            ocl->srcB = *bufferInputArray;
        }

        ocl->kernel = clCreateKernel(ocl->program, "Update_Weights_Buffers", &err);
        if (CL_SUCCESS != err)
        {
            LogError("Error: clCreateKernel returned %s\n", TranslateOpenCLError(err));
            return -1;
        }

        if (CL_SUCCESS != mSetKernelArguments(ocl, NULL, mDim, nDim, batchSize, learning_rate, 3))
        {
            return -1;
        }

        if (CL_SUCCESS != mExecuteMultiplyKernel(ocl, mDim, nDim)) //thankfully, the same function works for our Multiply_Deltas kernel too
        {
            return -1;
        }
    }
}

//Performing AxB, x is normal matrix multiplication
//A = weights, B = outputs for layer l-1, C = outputs for l, 
//where l is the current layer visited in this iteration of the forwardpassCpp loop
void multiplyIdKernelCpp(float* matrixA, float* matrixB, float* matrixC,
    const int mDim, const int pDim, const int nDim) {
    int idx;
    float temp;
    for (int i = 0; i < mDim; i++) {
        for (int j = 0; j < nDim; j++) {
            idx = i * nDim + j;
            temp = 0.0f;
            for (int k = 0; k < pDim; k++) {
                temp += matrixA[i * pDim + k] * matrixB[j + k * nDim];
            }
            matrixC[idx] = temp;
        }
    }
}

//Performing C = AxB and then elementwise sigmoid() on C
//x is normal matrix multiplication
//A = weights, B = outputs for layer l-1, C = outputs for l, 
//where l is the current layer visited in this iteration of the forwardpassCpp loop
void multiplySigmoidKernelCpp( float* matrixA, float* matrixB, float* matrixC,
    const int mDim, const int pDim, const int nDim) {
    int idx;
    float temp;
    for (int i = 0; i < mDim; i++) {
        for (int j = 0; j < nDim; j++) {
            idx = i * nDim + j;
            temp = 0.0f;
            for (int k = 0; k< pDim; k++) {
                temp += matrixA[i*pDim+k] * matrixB[j+k*nDim];
            }
            matrixC[idx] = (tanh(temp/2)+1)/2; //expressed sig as tanh because sig is not implemented in cmath
        }
    }
}

//Performing AxB, and then elementwise tanh() on C
//x is normal matrix multiplication
//A = weights, B = outputs for layer l-1, C = outputs for l, 
//where l is the current layer visited in this iteration of the forwardpassCpp loop
void multiplyTanhKernelCpp(float* matrixA, float* matrixB, float* matrixC,
    const int mDim, const int pDim, const int nDim) {
    int idx;
    float temp;
    for (int i = 0; i < mDim; i++) {
        for (int j = 0; j < nDim; j++) {
            idx = i * nDim + j;
            temp = 0.0f;
            for (int k = 0; k < pDim; k++) {
                temp += matrixA[i * pDim + k] * matrixB[j + k * nDim];
            }
            matrixC[idx] = tanh(temp);
        }
    }
}

//Performing AxB, and then elementwise ReLU() on C
//x is normal matrix multiplication
//A = weights, B = outputs for layer l-1, C = outputs for l,
//where l is the current layer visited in this iteration of the forwardpassCpp loop
void multiplyReLUKernelCpp(float* matrixA, float* matrixB, float* matrixC,
    const int mDim, const int pDim, const int nDim) {
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
            matrixC[idx] = fmax(temp,0.0f);
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
            matrixC[idx] = temp;
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
            matrixC[idx] = temp * matrixD[idx] * (1.0f - matrixD[idx]);
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
            matrixC[idx] = temp * (1.0f - pow(matrixD[idx], 2));
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
            matrixC[idx] = temp* (matrixD[idx] > 0.0 ? 1.0 : 0.0);
        }
    }
}
/////////////////////////////////////////////////////////////////////////////////////BP4///////////////////////////////////////
//Performing C' = AxB.T and then C = C - offset*C'
//x is normal matrix multiplication, * is element wise
//A = deltas, B = outputs, C = weights, offset = learning rate
void updateWeights(float* matrixA, float* matrixB, float* matrixC,
    const int mDim, const int pDim, const int nDim, const float offset)
{
    float temp;
    int idx1, idx2;
    for (int i = 0; i < mDim; i++) {
        for (int j = 0; j < nDim; j++) {
            temp = 0.0f;
            idx1 = i * pDim;
            idx2 = j * pDim;
            for (int k = 0; k < pDim; k++) {
                temp += matrixA[idx1 + k] * matrixB[idx2 + k];
            }
            temp = temp / (float)pDim;
            matrixC[i * nDim + j] = matrixC[i * nDim + j] - offset * temp;
        }
    }

    return;
}

int initializeparamsCpp(float*** weightsArray, float*** outputsArray, float*** deltasArray, float** inputArray, float** costs, int dimensions[], int batchSize, int iterations, int layers) {

    int optimizedSize = ((sizeof(float*) * layers - 1) / 64 + 1) * 64;
    *weightsArray = (float**)_aligned_malloc(optimizedSize, 4096); //array of memory objects, where each memory object is a buffer of weights between layers

    *outputsArray = (float**)_aligned_malloc(optimizedSize, 4096); //array of memory objects, where each memory object is a buffer image of outputs of layers

    *deltasArray = (float**)_aligned_malloc(optimizedSize, 4096); //array of memory objects, where each memory object is an image of deltas of layers

    int optimizedSizeIn = ((sizeof(float) * dimensions[0] * batchSize - 1) / 64 + 1) * 64;
    *inputArray = (float*)_aligned_malloc(optimizedSizeIn, 4096); //array of network input
    //std::cout << "Matrix input is \n";
    mGenerateMatrices(*inputArray, dimensions[0], batchSize);//TODO: have some code here to initiliaze inputArray with external training data
    if (NULL == *outputsArray || NULL == *deltasArray || NULL == *inputArray)
    {
        LogError("Error: _aligned_malloc failed to allocate buffers.\n");
        return -1;
    }
    int optimizedSizeCosts = ((sizeof(float) * iterations - 1) / 64 + 1) * 64;
    *costs = (float*)_aligned_malloc(optimizedSizeCosts, 4096);

    int optimizedSize1;
    int mDim, pDim;
    for (int x = 0; x < layers; ++x) {
        mDim = dimensions[x + 1];
        pDim = dimensions[x];

        optimizedSize1 = ((sizeof(float) * mDim * pDim - 1) / 64 + 1) * 64;
        (*weightsArray)[x] = (float*)_aligned_malloc(optimizedSize1, 4096);
        std::cout << "Weights of layer " << x << " are: \n";
        mGenerateMatrices((*weightsArray)[x], mDim, pDim);

        optimizedSize1 = ((sizeof(float) * mDim * batchSize - 1) / 64 + 1) * 64;
        (*outputsArray)[x] = (float*)_aligned_malloc(optimizedSize1, 4096);

        optimizedSize1 = ((sizeof(float) * mDim * batchSize - 1) / 64 + 1) * 64;
        (*deltasArray)[x] = (float*)_aligned_malloc(optimizedSize1, 4096);
    }

}

void printinn(float* matrixA, float* matrixB, float* matrixC,
    const int mDim, const int pDim, const int nDim) {
    int temp;
    for (int i = 0; i < mDim; i++) {
        for (int j = 0; j < pDim; j++) {
            std::cout << "i and j are " << i << " " << j <<" and res is "<< matrixA[i*pDim+j] <<'\n';
        }
    }
    std::cout << "switching to B";
    for (int i = 0; i < pDim; i++) {
        for (int j = 0; j < nDim; j++) {
            std::cout << "i and j are " << i << " " << j << " and res is " << matrixB[i * nDim + j] << '\n';
        }
    }

    std::cout << "switching to C";
    for (int i = 0; i < mDim; i++) {
        for (int j = 0; j < nDim; j++) {
            std::cout << "i and j are " << i << " " << j << " and res is " << matrixC[i * nDim + j] << '\n';
        }
    }
    return;
}

void forwardpassCpp(float** weightsArray,float** outputsArray, float* inputArray, int dimensions[], int* activationFunctions, int batchSize, int layers) {
    std::cout << "In forwardprop \n";
    float* srcA, * srcB;
    float* dstMem = inputArray;
    int mDim, pDim, nDim = batchSize, kernel;
    for (int x = 0; x < layers; ++x) {

        mDim = dimensions[x + 1];
        pDim = dimensions[x];

        srcA = weightsArray[x];
        srcB = dstMem;
        dstMem = outputsArray[x];

        kernel = activationFunctions[x];
        //printinn(srcA, srcB, dstMem, mDim, pDim, nDim);
        switch (kernel) {
        case 0:
            multiplyIdKernelCpp(srcA, srcB, dstMem, mDim, pDim, nDim);
            break;
        case 1:
            multiplySigmoidKernelCpp(srcA, srcB, dstMem, mDim, pDim, nDim);
            break;
        case 2:
            multiplyTanhKernelCpp(srcA, srcB, dstMem, mDim, pDim, nDim);
            break;
        case 3:
            multiplyReLUKernelCpp(srcA, srcB, dstMem, mDim, pDim, nDim);
            break;
        }
        //system("pause");
    }
}

void backpropCpp(float** weightsArray,float** outputsArray,float** deltasArray,float* inputArray, int* dimensions,
    int* activationFunctions, float* correctOutput, float* costs,int iter,int batchSize,int layers) {

    //std::cout << "In backprop for iter " << iter << " \n";

    int optimizedSizeNetworkOutput = ((sizeof(float) * batchSize - 1) / 64 + 1) * 64;
    float* resultPtr1 = (float*)_aligned_malloc(optimizedSizeNetworkOutput, 4096);
    float* networkOutput = (float*)_aligned_malloc(optimizedSizeNetworkOutput, 4096);
    for (int x = 0; x < batchSize; ++x) {

        networkOutput[x] = outputsArray[layers - 1][x];
        resultPtr1[x] = outputsArray[layers-1][x] ;
    }

    costs[iter] = MSECostFunction(correctOutput, resultPtr1, batchSize);

    //Then, let's compute the deltas, starting with the output delta 
    for (int x = 0; x < batchSize; ++x) {
        resultPtr1[x] = resultPtr1[x] - correctOutput[x];
    }

    if (activationFunctions[layers - 1] == 0) {
        for (int x = 0; x < batchSize; ++x) {
            networkOutput[x] = 1.0;
        }
    }
    else if (activationFunctions[layers - 1] == 1) {
        for (int x = 0; x < batchSize; ++x) {
            networkOutput[x] = networkOutput[x] * (1.0 - networkOutput[x]);
        }
    }
    else if (activationFunctions[layers - 1] == 2) {
        for (int x = 0; x < batchSize; ++x) {
            networkOutput[x] = 1.0f - (networkOutput[x]) * (networkOutput[x]);
        }
    }
    else if (activationFunctions[layers - 1] == 3) {
        for (int x = 0; x < batchSize; ++x) {
            if (networkOutput[x] == 0.0)
                networkOutput[x] = 0.0;
            else
                networkOutput[x] = 1.0;
        }
    }

    for (int x = 0; x < batchSize; ++x) {
        resultPtr1[x] = resultPtr1[x] * networkOutput[x];
    }
    for (int i = 0; i < batchSize; i++) {
        deltasArray[layers-1][i] = resultPtr1[i];
    }
    _aligned_free(resultPtr1);
    _aligned_free(networkOutput);

    //non-output deltas calculation loop
    float* srcA, * srcB, * dstMem;
    dstMem = deltasArray[layers - 1];
    float* outputs;
    int mDim, pDim, nDim = batchSize, kernel;;
    for (int x = layers - 1; x > 0; --x) {

        mDim = dimensions[x];
        pDim = dimensions[x + 1];
        srcA = weightsArray[x];
        srcB = dstMem;
        dstMem = deltasArray[x - 1];
        outputs = outputsArray[x - 1];

        kernel = activationFunctions[x];
        switch (kernel) {
        case 0:
            multiplyIdKernelCpp(srcA, srcB, dstMem, mDim, pDim, nDim);
            break;
        case 1:
            multiplySigmoidKernelCpp(srcA, srcB, dstMem, mDim, pDim, nDim);
            break;
        case 2:
            multiplyTanhKernelCpp(srcA, srcB, dstMem, mDim, pDim, nDim);
            break;
        case 3:
            multiplyReLUKernelCpp(srcA, srcB, dstMem, mDim, pDim, nDim);
            break;
        }
    }

    //perform weight updates now. We can potentially parallelize this fully even across all network layers
    //but for now it's only across the weights of each layer and then sequentially across layers
    float learning_rate = 0.06;
    pDim = batchSize;
    for (int x = layers - 1; x >= 0; --x) {
        //std::cout << "I'm in iteration " << x << " of the weight update loop \n";
        mDim = dimensions[x + 1];
        nDim = dimensions[x];
        srcA = deltasArray[x];
        dstMem = weightsArray[x];
        if (x != 0) {
            srcB = outputsArray[x - 1];
        }
        else {
            srcB = inputArray;
        }
        updateWeights(srcA, srcB, dstMem, mDim, pDim, nDim, learning_rate);
    }
}


void printWeights(ocl_args_d_t* ocl, cl_mem* buffersWeightsArray, int layers, int dimensions[], int batchSize) {
    std::cout << "Printing final weight values";

    cl_int err = CL_SUCCESS;

    cl_float* tempiboy;
    int mDim, pDim, optimizedSizeTempW;
    for (cl_uint x = 0; x < layers; ++x) {
        mDim = dimensions[x + 1];
        pDim = dimensions[x];
        int arraySize = mDim * pDim;
        optimizedSizeTempW = ((sizeof(cl_float) * mDim * pDim - 1) / 64 + 1) * 64;
        tempiboy = (cl_float*)_aligned_malloc(optimizedSizeTempW, 4096);

        err = clEnqueueReadBuffer(ocl->commandQueue, buffersWeightsArray[x], true, 0, sizeof(cl_float) * arraySize, tempiboy, 0, NULL, NULL);
        if (CL_SUCCESS != err)
        {
            LogError("Error: clEnqueueReadBuffer returned %s\n", TranslateOpenCLError(err));
        }


        for (cl_uint i = 0; i < arraySize; ++i)
        {
            std::cout << tempiboy[i] << " ";
            if ((i + 1) % pDim == 0) {
                std::cout << '\n';
            }
        }
        std::cout << '\n';
        _aligned_free(tempiboy);
    }
}


uchar** read_mnist_images(std::string full_path, int& number_of_images, int& image_size) {
    auto reverseInt = [](int i) {
        unsigned char c1, c2, c3, c4;
        c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
        return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
    };


    std::ifstream file(full_path, std::ios::binary);

    if (file.is_open()) {
        int magic_number = 0, n_rows = 0, n_cols = 0;

        file.read((char*)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);

        if (magic_number != 2051) throw std::runtime_error("Invalid MNIST image file!");

        file.read((char*)&number_of_images, sizeof(number_of_images)), number_of_images = reverseInt(number_of_images);
        file.read((char*)&n_rows, sizeof(n_rows)), n_rows = reverseInt(n_rows);
        file.read((char*)&n_cols, sizeof(n_cols)), n_cols = reverseInt(n_cols);

        image_size = n_rows * n_cols;

        uchar** _dataset = new uchar * [number_of_images];
        for (int i = 0; i < number_of_images; i++) {
            _dataset[i] = new uchar[image_size];
            file.read((char*)_dataset[i], image_size);
        }
        return _dataset;
    }
    else {
        throw std::runtime_error("Cannot open file `" + full_path + "`!");
    }
}

uchar* read_mnist_labels(std::string full_path, int& number_of_labels) {
    auto reverseInt = [](int i) {
        unsigned char c1, c2, c3, c4;
        c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
        return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
    };

    typedef unsigned char uchar;

    std::ifstream file(full_path, std::ios::binary);

    if (file.is_open()) {
        int magic_number = 0;
        file.read((char*)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);

        if (magic_number != 2049) throw std::runtime_error("Invalid MNIST label file!");

        file.read((char*)&number_of_labels, sizeof(number_of_labels)), number_of_labels = reverseInt(number_of_labels);

        uchar* _dataset = new uchar[number_of_labels];
        for (int i = 0; i < number_of_labels; i++) {
            file.read((char*)&_dataset[i], 1);
        }
        return _dataset;
    }
    else {
        throw std::runtime_error("Unable to open file `" + full_path + "`!");
    }
}

//cl_uint minibatchGD() {
//
//
//
//}

int _tmain(int argc, TCHAR* argv[])
{
    cl_int err;
    ocl_args_d_t ocl;
    cl_device_type deviceType = CL_DEVICE_TYPE_GPU;
    //LARGE_INTEGER perfFrequency;
    //LARGE_INTEGER performanceCountNDRangeStart;
    //LARGE_INTEGER performanceCountNDRangeStop;

    cl_uint mDim = 1;
    cl_uint pDim;
    cl_uint nDim;
    //A FEW EXPLANATIONS FOR HOW THIS WORKS
    //first, we have our outer loop that performs for a set number of iterations
    //in each iterations, we perform 3 different loops
    //the 1st loop, corresponds to the inference phase (forward pass).
    //The 2nd loop (and the little bit of code before it) calculates the deltas, i.e. the
    //partial derivatives of the cost function wrt each node's pre-activation-function output.
    //This is necessary because the way we calculate the partial derivatives of the weights is using
    //the chain rule, and one of the 2 terms that comes up is this one (the next one is calculated
    //in the next loop).
    //The 3rd loop calculates the second term of the partial derivatives of the weights and updates
    //the weights. The reason I created both a delta calculation loop and a weight update loop,
    //is that even though the delta calculation has to happen layer by layer (since previous layer deltas
    //are used to calculate next layer ones), the weight updates can be completely parallelized across
    // both nodes (as with deltas) and layers. This means I can just replace the 3rd loop with one
    //clEnqueueNDRangeKernel invocation (haven't done this yet, but will in the future).

    const int batchSize = 256;
    const int layers = 2; //We don't count input as a layer
    int dimAr[layers + 1] = {784, 1024, 10}; //last layer should always be set to 1 for regression
    //cl_float correctOutput[batchSize] = { 1.0,2.0,3.0,4.0,5.0,6.0,7.0,-77.0,-8.0,8.5,9.2,-10.0}; //The desired output
    const int numAF = 4; //num of activation functions
    char* activationFunctionKernelNames[numAF] = { "Multiply_Buffer_Identity",
    "Multiply_Buffer_Sigmoid","Multiply_Buffer_Tanh","Multiply_Buffer_ReLU" };
    char* activationFunctionDeltaKernelNames[numAF] = { "Multiply_Deltas_Buffers_Identity",
    "Multiply_Deltas_Buffers_Sigmoid","Multiply_Deltas_Buffers_Tanh","Multiply_Deltas_Buffers_ReLU" };
    cl_kernel activationFunctionKernels[numAF], activationFunctionDeltaKernels[numAF];
    cl_kernel activationFunctionKernelsSimple[numAF], activationFunctionDeltaKernelsSimple[numAF];

    //for (int x = 0; x < batchSize; ++x) {
    //    std::cout << "Correct output term "<<x<<" is"<< correctOutput[x] << '\n';
    //}
    int activationFunctions[layers] = {3,0}; // 0 for identity, 1 for tanh, 2 for sigmoid, 3 for ReLU

    //initialize Open CL objects (context, queue, etc.)
    if (CL_SUCCESS != SetupOpenCL(&ocl, deviceType))
    {
        return -1;
    }

    // Create and build the OpenCL program
    if (CL_SUCCESS != CreateAndBuildProgram(&ocl))
    {
        return -1;
    }

    
    for (int x = 0; x < numAF; ++x) {
        activationFunctionKernels[x] = clCreateKernel(ocl.program, activationFunctionKernelNames[x], &err);
        if (CL_SUCCESS != err)
        {
            LogError("Error: clCreateKernel returned %s\n", TranslateOpenCLError(err));
            return -1;
        }
        activationFunctionDeltaKernels[x] = clCreateKernel(ocl.program, activationFunctionDeltaKernelNames[x], &err);
        if (CL_SUCCESS != err)
        {
            LogError("Error: clCreateKernel returned %s\n", TranslateOpenCLError(err));
            return -1;
        }
        activationFunctionKernelsSimple[x] = clCreateKernel(ocl.programSimple, activationFunctionKernelNames[x], &err);
        if (CL_SUCCESS != err)
        {
            LogError("Error: clCreateKernel returned %s\n", TranslateOpenCLError(err));
            return -1;
        }
        activationFunctionDeltaKernelsSimple[x] = clCreateKernel(ocl.programSimple, activationFunctionDeltaKernelNames[x], &err);
        if (CL_SUCCESS != err)
        {
            LogError("Error: clCreateKernel returned %s\n", TranslateOpenCLError(err));
            return -1;
        }
    }

    int trainImageSize, valImageSize, numTrainImages, numTrainLabels, numValImages, numValLabels;
    uchar** trainingDataset = read_mnist_images("C:\\Users\\george cabon x1\\source\\repos\\GPUOpenCLProjectforWindows2\\GPUOpenCLProjectforWindows2\\MNIST\\train-images.idx3-ubyte", numTrainImages, trainImageSize);
    uchar* trainingLabels = read_mnist_labels("C:\\Users\\george cabon x1\\source\\repos\\GPUOpenCLProjectforWindows2\\GPUOpenCLProjectforWindows2\\MNIST\\train-labels.idx1-ubyte", numTrainLabels);
    
    uchar** valDataset = read_mnist_images("C:\\Users\\george cabon x1\\source\\repos\\GPUOpenCLProjectforWindows2\\GPUOpenCLProjectforWindows2\\MNIST\\t10k-images.idx3-ubyte", numValImages, valImageSize);
    uchar* valLabels = read_mnist_labels("C:\\Users\\george cabon x1\\source\\repos\\GPUOpenCLProjectforWindows2\\GPUOpenCLProjectforWindows2\\MNIST\\t10k-labels.idx1-ubyte", numValLabels);

    std::cout << "There are " << numTrainImages << " with size " << valImageSize << '\n';
    std::cout << "First image label is " << (cl_float)trainingLabels[0] << '\n';
    for (int i = 0; i < 28; ++i) {
        for (int j = 0; j < 28; ++j) {
            if (trainingDataset[0][28 * i + j] > 100) {
                std::cout << 1;
            }
            else {
                std::cout << 0;
            }
        }
        std::cout << '\n';
    }

    int epochs = 30;
    int classes = 10;

    //MAIN LOOP
    
    float** weightsArray;
    cl_mem* weightBuffers, * outputBuffers, * deltaBuffers, inputBuffer;
    cl_float* correctOutput;
    cl_float* costs;

    initializeparamsClassifier(&ocl, &weightBuffers, &outputBuffers, &deltaBuffers, &inputBuffer, &costs, dimAr, batchSize, epochs, layers, &correctOutput, trainingDataset, trainingLabels);

    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::cout << "epoch "<<epoch<<" hereeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee \n";
        std::cout << "ENTERING OPENCL FORWARD" << "\n";
        forwardpassClassifier(&ocl, weightBuffers, outputBuffers, &inputBuffer, dimAr, activationFunctions, activationFunctionKernels, activationFunctionKernelsSimple, batchSize, layers);
        std::cout << "ENTERING OPENCL BACKWARD" << "\n";
        backpropClassifier(&ocl, weightBuffers, outputBuffers, deltaBuffers, &inputBuffer, dimAr, activationFunctions, activationFunctionDeltaKernels, activationFunctionDeltaKernelsSimple, correctOutput, costs, epoch, batchSize, layers, classes);
    }

    for (int i = 0; i < epochs; i++) {
        std::cout << costs[i] << '\n';
    }

    validationClassifier(&ocl, weightBuffers, outputBuffers, &inputBuffer, dimAr, activationFunctions, activationFunctionKernels,
        activationFunctionKernelsSimple, layers, classes, valDataset, valLabels, numValImages);

    std::cout << "finally done";
    system("pause");

    //C++ LOOPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP

    /*float** weightsArray, ** outputsArray, ** deltasArray, *inputArray;
    float* costs;
    //initializeparamsCpp(&weightsArray, &outputsArray, &deltasArray, &inputArray, &costs1, dimAr, batchSize, iterations, layers);
    initializeparamsCpp(&weightsArray, &outputsArray, &deltasArray, &inputArray, &costs, dimAr, batchSize, iterations, layers);

    for (int iter = 0; iter < iterations; ++iter) {
        std::cout << "iteration " << iter << " here \n";
        std::cout << "ENTERING C++ FORWARD" << "\n";
        forwardpassCpp(weightsArray, outputsArray, inputArray, dimAr, activationFunctions, batchSize, layers);
        std::cout << "ENTERING C++ BACKWARD" << "\n";
        backpropCpp(weightsArray, outputsArray, deltasArray, inputArray, dimAr, activationFunctions, correctOutput, costs, iter, batchSize, layers);
    }

    for (int i = 0; i < iterations; i++) {
        std::cout << costs[i] << '\n';
    }
    std::cout << "finally done";
    system("pause");*/

    //BASIC TESTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT
    /*cl_uint optimizedSizeTempA = ((sizeof(cl_float) * mDim * pDim - 1) / 64 + 1) * 64;
    cl_uint optimizedSizeTempB = ((sizeof(cl_float) * pDim * nDim - 1) / 64 + 1) * 64;
    cl_uint optimizedSizeTempC = ((sizeof(cl_float) * mDim * nDim - 1) / 64 + 1) * 64;
    cl_float* matrixAar = (cl_float*)_aligned_malloc(optimizedSizeTempA, 4096);
    cl_float* matrixBar = (cl_float*)_aligned_malloc(optimizedSizeTempB, 4096);
    cl_float* matrixCar = (cl_float*)_aligned_malloc(optimizedSizeTempC, 4096);
    mGenerateMatrices(matrixAar, mDim, pDim);
    mGenerateMatrices(matrixBar, pDim, nDim);
    //mGenerateMatrices(matrixCar, mDim, nDim);
    
    float(*matrixDar)[dim] = new float[dim][dim];
    float(*matrixEar)[dim] = new float[dim][dim];
    float(*matrixFar)[dim] = new float[dim][dim];
    mGenerateMatrices2D(matrixDar, dim, dim);
    mGenerateMatrices2D(matrixEar, dim, dim);

    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            matrixFar[i][j] = 0.0;
        }
    }

    //MAYBE ADD CODE TO MAKE SURE matrixCar starts with 0 values

    using namespace std::chrono;
    auto start = high_resolution_clock::now();
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            for (int k = 0; k < dim; k++) {
                matrixFar[i][j] += (matrixDar[i][k]) * (matrixEar[k][j]); //MAYBE IT'S SLOW DUE TO MULTIPLICATIONS
            }
        }
    }
    auto elapsed = std::chrono::high_resolution_clock::now() - start;
    long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    std::cout << microseconds << " microseconds since epoch\n";*/
    //matrixmulttestCPU();
    //GPU TESTINGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG
/*int total = 0;
    for (int zi = 0; zi < 50; zi++) {
        int comdim = 512;
        mDim = comdim, pDim = comdim, nDim = comdim;
        cl_uint optimizedSizeTempAA = ((sizeof(cl_float) * mDim * pDim - 1) / 64 + 1) * 64;
        cl_uint optimizedSizeTempBB = ((sizeof(cl_float) * pDim * nDim - 1) / 64 + 1) * 64;
        cl_uint optimizedSizeTempCC = ((sizeof(cl_float) * mDim * nDim - 1) / 64 + 1) * 64;
        cl_float* matrixAar = (cl_float*)_aligned_malloc(optimizedSizeTempAA, 4096);
        cl_float* matrixBar = (cl_float*)_aligned_malloc(optimizedSizeTempBB, 4096);
        cl_float* matrixCar = (cl_float*)_aligned_malloc(optimizedSizeTempCC, 4096);
        cl_float* matrixDar = (cl_float*)_aligned_malloc(optimizedSizeTempCC, 4096);
        mGenerateMatrices(matrixAar, mDim, pDim);
        mGenerateMatrices(matrixBar, pDim, nDim);


        // Create first buffer based on host memory inputA
        cl_mem matrixA = clCreateBuffer(ocl.context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * mDim * pDim, matrixAar, &err);
        cl_mem matrixB = clCreateBuffer(ocl.context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * pDim * nDim, matrixBar, &err);
        cl_mem matrixC = clCreateBuffer(ocl.context, CL_MEM_READ_WRITE, sizeof(cl_float) * mDim * nDim, NULL, &err);
        cl_mem matrixD = clCreateBuffer(ocl.context, CL_MEM_READ_WRITE, sizeof(cl_float) * mDim * nDim, NULL, &err);
        //clCreateBuffer(ocl.context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * mDim * nDim, matrixCar, &err);
        if (CL_SUCCESS != err)
        {
            LogError("Error: matrix creation failed with %s\n", TranslateOpenCLError(err));
            return err;
        }

        using namespace std::chrono;

        ocl.srcA = matrixA;
        ocl.srcB = matrixB;
        ocl.dstMem = matrixC;

        ocl.kernel = clCreateKernel(ocl.program, "Matrix_Multiply_Kernel_3", &err);;
        // Passing arguments into OpenCL kernel.
        if (CL_SUCCESS != mSetKernelArguments(&ocl, NULL, mDim, pDim, nDim, 0.0, 1))
        {
            return -1;
        }

        auto start1 = high_resolution_clock::now();
        int TS = 16;
        int WPT = 8;
        const size_t global[2] = { comdim, comdim / WPT };
        const size_t local[2] = { TS,TS / WPT };
        if (CL_SUCCESS != mExecuteMultiplyKernelCustom(&ocl, global, local))
        {
            return -1;
        }
        auto elapsed1 = std::chrono::high_resolution_clock::now() - start1;
        long long microseconds1 = std::chrono::duration_cast<std::chrono::microseconds>(elapsed1).count();
        std::cout << microseconds1 << " microseconds since epoch\n";
        total += microseconds1 / 50;

        err = clEnqueueReadBuffer(ocl.commandQueue, matrixC, true, 0, sizeof(cl_float) * mDim * nDim, matrixCar, 0, NULL, NULL);
        if (CL_SUCCESS != err)
        {
            LogError("Error: clEnqueueReadBuffer returned %s\n", TranslateOpenCLError(err));
        }
    }
    std::cout << "total average is " << total;
    system("pause");*/

    /*for (cl_uint i = 0; i < mDim * nDim; ++i){
        std::cout << matrixCar[i] << " ";
        if ((i + 1) % nDim == 0) {
            std::cout << '\n';
        }
    }

    ocl.srcA = matrixA;
    ocl.srcB = matrixB;
    ocl.dstMem = matrixD;

    ocl.kernel = clCreateKernel(ocl.program, "Matrix_Multiply_Kernel_1", &err);;
    // Passing arguments into OpenCL kernel.
    if (CL_SUCCESS != mSetKernelArguments(&ocl, NULL, mDim, pDim, nDim, 0.0, 1))
    {
        return -1;
    }

    auto start3 = high_resolution_clock::now();
    if (CL_SUCCESS != mExecuteMultiplyKernel(&ocl, mDim, nDim))
    {
        return -1;
    }
    auto elapsed3 = std::chrono::high_resolution_clock::now() - start3;
    long long microseconds3 = std::chrono::duration_cast<std::chrono::microseconds>(elapsed3).count();
    std::cout << microseconds3 << " microseconds since epoch\n";

    err = clEnqueueReadBuffer(ocl.commandQueue, matrixD, true, 0, sizeof(cl_float) * mDim * nDim, matrixDar, 0, NULL, NULL);
    if (CL_SUCCESS != err)
    {
        LogError("Error: clEnqueueReadBuffer returned %s\n", TranslateOpenCLError(err));
    }


    for (cl_uint i = 0; i < mDim*nDim; ++i)
    {
        std::cout << matrixCar[i] << " ";
        if ((i + 1) % nDim == 0) {
            std::cout << '\n';
        }
    }*/

    /*
    cl_uint optimizedSize = ((sizeof(cl_mem) * layers - 1) / 64 + 1) * 64;
    cl_mem* buffersWeightsArray = (cl_mem*)_aligned_malloc(optimizedSize, 4096); //array of memory objects, where each memory object is an image of weights between layers

    cl_mem* buffersOutsArray = (cl_mem*)_aligned_malloc(optimizedSize, 4096); //array of memory objects, where each memory object is an image of outputs of layers

    cl_mem* buffersDeltasArray = (cl_mem*)_aligned_malloc(optimizedSize, 4096); //array of memory objects, where each memory object is an image of deltas of layers

    cl_uint optimizedSizeIn = ((sizeof(cl_float) * Dims[0] * batchSize - 1) / 64 + 1) * 64;
    cl_float* inArray = (cl_float*)_aligned_malloc(optimizedSizeIn, 4096); //array of network input
    //std::cout << "Matrix input is \n";
    mGenerateMatrices(inArray, Dims[0], batchSize);//TODO: have some code here to initiliaze inputArray with external training data
    
    if (NULL == buffersWeightsArray || NULL == buffersOutsArray || NULL == buffersDeltasArray ||NULL == inArray)
    {
        LogError("Error: _aligned_malloc failed to allocate buffers.\n");
        return -1;
    }

    cl_mem bufferInArray = clCreateBuffer(ocl.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_float)*Dims[0]*batchSize, inArray, &err);
    if (CL_SUCCESS != err)
    {
        LogError("Error: clCreateImage for srcA returned %s\n", TranslateOpenCLError(err));
        return err;
    }

    cl_uint optimizedSizeTempOutAndDelta;
    cl_uint optimizedSizeTempW;
    cl_float* tempWeightArray;
    
    for (cl_uint x = 0; x < layers; ++x) {
            mDim = Dims[x + 1];
            pDim = Dims[x];
            //optimizedSizeTempOutAndDelta = ((sizeof(cl_int) * mDim - 1) / 64 + 1) * 64;
            optimizedSizeTempW = ((sizeof(cl_float) * mDim * pDim - 1) / 64 + 1) * 64;
            tempWeightArray = (cl_float*)_aligned_malloc(optimizedSizeTempW, 4096);
            std::cout << "Weights of layer " << x << " are: \n";
            mGenerateMatrices(tempWeightArray, mDim, pDim);

            // Create first buffer based on host memory inputA
            buffersWeightsArray[x] = clCreateBuffer(ocl.context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * mDim * pDim, tempWeightArray, &err);
            if (CL_SUCCESS != err)
            {
                LogError("Error: clCreateImage for srcA returned %s\n", TranslateOpenCLError(err));
                return err;
            }

            buffersOutsArray[x] = clCreateBuffer(ocl.context, CL_MEM_READ_WRITE, sizeof(cl_float) * mDim * batchSize, NULL, &err);
            if (CL_SUCCESS != err)
            {
                LogError("Error: clCreateImage for srcA returned %s\n", TranslateOpenCLError(err));
                return err;
            }

            buffersDeltasArray[x] = clCreateBuffer(ocl.context, CL_MEM_READ_WRITE, sizeof(cl_float) * mDim * batchSize, NULL, &err);
            if (CL_SUCCESS != err)
            {
                LogError("Error: clCreateImage for srcA returned %s\n", TranslateOpenCLError(err));
                return err;
            }
            _aligned_free(tempWeightArray);
    }*/

    //auto duration = duration_cast<microseconds>(stop - start);
    //std::cout << duration.count() << '\n';

    //err = clEnqueueReadBuffer(ocl.commandQueue, matrixC, true, 0, sizeof(cl_float) *mDim*nDim, matrixCar, 0, NULL, NULL);
    //if (CL_SUCCESS != err)
    //{
    //    LogError("Error: clEnqueueReadBuffer returned %s\n", TranslateOpenCLError(err));
    //}
    //for (cl_uint i = 0; i < mDim*nDim; ++i)
    //{
    //    std::cout << matrixCar[i] << " ";
    //    if ((i + 1) % nDim == 0) {
    //        std::cout << '\n';
    //    }
    //}




    /*int iterations = 250;
    int optimizedSizeCosts = ((sizeof(cl_float) * iterations - 1) / 64 + 1) * 64;
    cl_float* costs = (cl_float*)_aligned_malloc(optimizedSizeCosts, 4096);
    using namespace std::chrono;
    auto start = high_resolution_clock::now();



    
    for (int iter = 0; iter < iterations; iter++) {
        std::cout << "iteration here \n";
        ocl.dstMem = bufferInArray;
        nDim = batchSize;
        for (int x = 0; x < layers; ++x) {

            mDim = Dims[x + 1];
            pDim = Dims[x];

            ocl.srcA = buffersWeightsArray[x];
            ocl.srcB = ocl.dstMem;
            ocl.dstMem = buffersOutsArray[x];

            // Program consists of kernels.
            // Each kernel can be called (enqueued) from the host part of OpenCL application.
            // To call the kernel, you need to create it from existing program.
            ocl.kernel = activationFunctionKernels[ActivationFunctions[x]];
            // Passing arguments into OpenCL kernel.
            if (CL_SUCCESS != mSetKernelArguments(&ocl, NULL, mDim, pDim, nDim, 0.0, 1))
            {
                return -1;
            }
            //if (CL_SUCCESS != mExecuteMultiplyKernel(&ocl, mDim, nDim))
            if (CL_SUCCESS != mExecuteMultiplyKernel(&ocl, mDim, nDim))
            {
                return -1;
            }
        }

        //Using the mean squared error (MSE) cost function, we apply backpropagation
        //First, let's compute the cost function
        //TODO: CHANGE THIS TO ACCOMMODATE NON-SINGLE OUTPUT NODE ARCHITECTURES
        cl_uint optimizedSizeNetworkOutput = ((sizeof(cl_float)*batchSize - 1) / 64 + 1) * 64;
        cl_float* resultPtr1 = (cl_float*)_aligned_malloc(optimizedSizeNetworkOutput, 4096);
        cl_float* networkOutput = (cl_float*)_aligned_malloc(optimizedSizeNetworkOutput, 4096);
        err = clEnqueueReadBuffer(ocl.commandQueue, ocl.dstMem, true, 0, sizeof(cl_float)*batchSize, resultPtr1, 0, NULL, NULL);
        if (CL_SUCCESS != err)
        {
            LogError("Error: clEnqueueReadBuffer returned %s\n", TranslateOpenCLError(err));
        }
        for (int x = 0; x < batchSize; ++x) {
            networkOutput[x] = resultPtr1[x];
        }
        costs[iter] = MSECostFunction(correctOutput, resultPtr1, batchSize);

        //Then, let's compute the deltas, starting with the output delta 
        for (int x = 0; x < batchSize; ++x) {
            resultPtr1[x] = resultPtr1[x] - correctOutput[x];
        }

        if (ActivationFunctions[layers - 1] == 0) {
            for (int x = 0; x < batchSize; ++x) {
                networkOutput[x] = 1.0;
            }
        }
        else if (ActivationFunctions[layers - 1] == 1) {
            for (int x = 0; x < batchSize; ++x) {
                networkOutput[x] = networkOutput[x] * (1.0 - networkOutput[x]);
            }
        }
        else if (ActivationFunctions[layers - 1] == 2) {
            for (int x = 0; x < batchSize; ++x) {
                networkOutput[x] = 1.0 - networkOutput[x] * networkOutput[x];
            }
        }
        else if (ActivationFunctions[layers - 1] == 3) {
            for (int x = 0; x < batchSize; ++x) {
                if (networkOutput[x] == 0.0)
                    networkOutput[x] = 0.0;
                else
                    networkOutput[x] = 1.0;
            }
        }
        for (int x = 0; x < batchSize; ++x) {
            // std::cout << "Output delta function term "<<x<<" is: " << networkOutput[x] << '\n';
            // std::cout << "Output delta pre function term " << x << "is: " << resultPtr1[x] << '\n';
        }

        for (int x = 0; x < batchSize; ++x) {
            resultPtr1[x] = resultPtr1[x] * networkOutput[x];
        }
        for (int x = 0; x < batchSize; ++x) {
            // std::cout << "Output delta term "<<x<<" is: " << resultPtr1[x] << '\n';
        }
        err = clEnqueueWriteBuffer(ocl.commandQueue, buffersDeltasArray[layers - 1], true, 0, sizeof(cl_float) * batchSize, resultPtr1, 0, NULL, NULL);
        if (CL_SUCCESS != err)
        {
            LogError("Error: clEnqueueReadBuffer returned %s\n", TranslateOpenCLError(err));
        }
        _aligned_free(resultPtr1);
        _aligned_free(networkOutput);

        //non-output deltas calculation loop
        ocl.dstMem = buffersDeltasArray[layers - 1];
        cl_mem outputs;
        nDim = batchSize;
        for (int x = layers - 1; x > 0; --x) {

            mDim = Dims[x];
            pDim = Dims[x + 1];
            ocl.srcA = buffersWeightsArray[x];
            ocl.srcB = ocl.dstMem;
            ocl.dstMem = buffersDeltasArray[x - 1];
            outputs = buffersOutsArray[x - 1];

            // Program consists of kernels.
            // Each kernel can be called (enqueued) from the host part of OpenCL application.
            // To call the kernel, you need to create it from existing program.
            ocl.kernel = activationFunctionDeltasKernels[ActivationFunctions[x]];
            // Passing arguments into OpenCL kernel.
            if (CL_SUCCESS != mSetKernelArguments(&ocl, &outputs, mDim, pDim, nDim, 0.0, 2))
            {
                return -1;
            }
            bool queueProfilingEnable = true;
            if (queueProfilingEnable)
                QueryPerformanceCounter(&performanceCountNDRangeStart);
            // Execute (enqueue) the kernel
            if (CL_SUCCESS != mExecuteMultiplyKernel(&ocl, mDim, nDim)) //thankfully, the same function works for our Multiply_Deltas kernel too
            {
                return -1;
            }
            if (queueProfilingEnable)
                QueryPerformanceCounter(&performanceCountNDRangeStop);
            //std::cout << "Outputing delta from layer " << x << '\n';
        }

        //perform weight updates now. We can potentially parallelize this fully even across all network layers
        //but for now it's only across the weights of each layer and then sequentially across layers
        cl_float learning_rate = 0.03;
        for (int x = layers - 1; x >= 0; --x) {
            //std::cout << "I'm in iteration " << x << " of the weight update loop \n";
            mDim = Dims[x + 1];
            nDim = Dims[x];
            ocl.srcA = buffersDeltasArray[x];
            ocl.dstMem = buffersWeightsArray[x];
            if (x != 0) {
                ocl.srcB = buffersOutsArray[x - 1];
            }
            else {
                ocl.srcB = bufferInArray;
            }

            ocl.kernel = clCreateKernel(ocl.program, "Update_Weights_Buffers", &err);
            if (CL_SUCCESS != err)
            {
                LogError("Error: clCreateKernel returned %s\n", TranslateOpenCLError(err));
                return -1;
            }

            if (CL_SUCCESS != mSetKernelArguments(&ocl, NULL, mDim, nDim, batchSize, learning_rate, 3))
            {
                return -1;
            }

            bool queueProfilingEnable = true;
            if (queueProfilingEnable)
                QueryPerformanceCounter(&performanceCountNDRangeStart);
            // Execute (enqueue) the kernel

            if (CL_SUCCESS != mExecuteMultiplyKernel(&ocl, mDim, nDim)) //thankfully, the same function works for our Multiply_Deltas kernel too
            {
                return -1;
            }
            if (queueProfilingEnable)
                QueryPerformanceCounter(&performanceCountNDRangeStop);
            //std::cout << "Outputing weights from layer " << x << '\n';
            if (iter == 249) {
                cl_float* tempiboy2;
                int tempo1 = Dims[x + 1];
                int tempo2 = Dims[x];
                int arraySize = tempo1*tempo2;
                optimizedSizeTempW = ((sizeof(cl_float) * tempo1 *tempo2 -1) / 64 + 1) * 64;
                tempiboy2 = (cl_float*)_aligned_malloc(optimizedSizeTempW, 4096);

                err = clEnqueueReadBuffer(ocl.commandQueue, ocl.dstMem, true, 0, sizeof(cl_float) * arraySize, tempiboy2, 0, NULL, NULL);
                if (CL_SUCCESS != err)
                {
                    LogError("Error: clEnqueueReadBuffer returned %s\n", TranslateOpenCLError(err));
                }
                for (cl_uint i = 0; i < arraySize; ++i)
                {
                    std::cout << tempiboy2[i] << " ";
                    if ((i + 1) % tempo2 == 0) {
                        std::cout << '\n';
                    }
                }
                std::cout << '\n';
                _aligned_free(tempiboy2);
            }
        }
    }
    using namespace std::chrono;

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    std::cout << duration.count() << '\n';
        
    for (int i = 0; i < iterations; i++) {
            std::cout << costs[i] << '\n';
    }

    std::cout << "Printing final weight values";

    cl_float* tempiboy;
    for (cl_uint x = 0; x < layers; ++x) {
        mDim = Dims[x + 1];
        pDim = Dims[x];
        int arraySize = mDim * pDim;
        optimizedSizeTempW = ((sizeof(cl_float) * mDim * pDim - 1) / 64 + 1) * 64;
        tempiboy =  (cl_float*)_aligned_malloc(optimizedSizeTempW, 4096);

        err = clEnqueueReadBuffer(ocl.commandQueue, buffersWeightsArray[x], true, 0, sizeof(cl_float) * arraySize, tempiboy, 0, NULL, NULL);
        if (CL_SUCCESS != err)
        {
            LogError("Error: clEnqueueReadBuffer returned %s\n", TranslateOpenCLError(err));
        }
        

        for (cl_uint i = 0; i < arraySize; ++i)
        {
            std::cout << tempiboy[i] << " ";
            if ((i + 1) % pDim == 0) {
                std::cout << '\n';
            }
        }
        std::cout << '\n';
        _aligned_free(tempiboy);
    }


    std::cout << "Printed them";

    //Deallocate dynamic memory and release memory objects
    for (cl_uint x = 0; x < layers; ++x) {
            //std::cout << "releasing obj num" << x;
            clReleaseMemObject(buffersOutsArray[x]);
            clReleaseMemObject(buffersWeightsArray[x]);
            clReleaseMemObject(buffersDeltasArray[x]);
    }

    for (int x = 0; x < numAF; ++x) {
            clReleaseKernel(activationFunctionKernels[x]);
            clReleaseKernel(activationFunctionDeltasKernels[x]);
    }
    //_aligned_free(costs);
    _aligned_free(buffersOutsArray);
    _aligned_free(buffersWeightsArray);
    _aligned_free(buffersDeltasArray);
    clReleaseMemObject(bufferInArray);
                   
    //cl_platform_id platformId = FindOpenCLPlatform("Intel", deviceType);
    //PrintDeviceIDs(platformId);
    std::cout << "please ereach here \n";
    system("pause");
    return 0;*/
}


    //    mdim = 1000;
    //    pdim = 1000;
    //    ndim = 1000;

    //    cl_uint optimizedsizetempw1 = ((sizeof(cl_float) * mdim * pdim - 1) / 64 + 1) * 64;
    //    cl_uint optimizedsizetempw2 = ((sizeof(cl_float) * ndim * pdim - 1) / 64 + 1) * 64;
    //    cl_float* tempweightarray1 = (cl_float*)_aligned_malloc(optimizedsizetempw1, 4096);
    //    cl_float* tempweightarray2 = (cl_float*)_aligned_malloc(optimizedsizetempw2, 4096);
    //    //std::cout << "weights of layer " << x << " are: \n";


    //    // create first image based on host memory inputa
    //    cl_mem buffersweightsarray3 = clcreatebuffer(ocl.context, cl_mem_read_write, sizeof(cl_float) * mdim * ndim, null, &err);

    //    if (cl_success != setupopencl(&ocl, devicetype))
    //    {
    //        return -1;
    //    }

    //    // create and build the opencl program
    //    if (cl_success != createandbuildprogram(&ocl))
    //    {
    //        return -1;
    //    }

    //    cl_mem buffersweightsarray1;
    //    cl_mem buffersweightsarray2;
    //    int iterations = 1;
    //    auto start = high_resolution_clock::now();
    //    for (int x = 0; x < iterations; ++x) {
    //        mgeneratematrices(tempweightarray1, mdim, pdim);
    //        mgeneratematrices(tempweightarray2, pdim, ndim);
    //        buffersweightsarray1 = clcreatebuffer(ocl.context, cl_mem_read_write | cl_mem_copy_host_ptr, sizeof(cl_float) * mdim * pdim, tempweightarray1, &err);
    //        buffersweightsarray2 = clcreatebuffer(ocl.context, cl_mem_read_write | cl_mem_copy_host_ptr, sizeof(cl_float) * pdim * ndim, tempweightarray2, &err);
    //        ocl.srca = buffersweightsarray1;
    //        ocl.srcb = buffersweightsarray2;
    //        ocl.dstmem = buffersweightsarray3;

    //        ocl.kernel = clcreatekernel(ocl.program, "multiply_buffer_identity", &err);
    //        if (cl_success != msetkernelarguments(&ocl, null, mdim, pdim, ndim, 0.0, 1))
    //        {
    //            return -1;
    //        }
    //        if (cl_success != mexecutemultiplykernel(&ocl, mdim, 1))
    //        {
    //            return -1;
    //        }
    //        clreleasememobject(buffersweightsarray1);
    //        clreleasememobject(buffersweightsarray2);
    //    }
    //}