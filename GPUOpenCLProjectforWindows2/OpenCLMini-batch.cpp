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

#include "C:\Users\george cabon x1\source\repos\GPUOpenCLProjectforWindows2\GPUOpenCLProjectforWindows2\NaiveC++Mini-batch\NaiveC++Mini-batch.h"

using namespace std::chrono;
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
    if (srcA)
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
    }
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

    //cl_ulong size;
    //clGetDeviceInfo(ocl->device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &size, 0);
    //if (CL_SUCCESS != err)
    //{
    //    LogError("Error: clGetDeviceInfo() to get CL_DEVICE_VERSION returned %s.\n", TranslateOpenCLError(err));
    //    return err;
    //}
    //std::cout << "TOTAL LOCAL MEMORY IS " << size << '\n';
    //system("pause");

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
 

void mGenerateMatrices(cl_float* inputArray, cl_uint height, cl_uint width)
{
    cl_float temp = 0;

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
    err = ReadSourceFromFile("kernel3.cl", &source, &srcSize);
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
    ocl->program = clCreateProgramWithSource(ocl->context, 1, (const char**)&source, &srcSize, &err);
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
cl_uint mSetKernelArguments(ocl_args_d_t *ocl, cl_mem* matrixD, cl_uint mDim, cl_uint pDim, cl_uint nDim, cl_float learningRate, cl_uint kernel)
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
       LogError("Error: Failed to set argument mDim, returned %s\n", TranslateOpenCLError(err));
       return err;
    }

    err = clSetKernelArg(ocl->kernel, 4, sizeof(cl_uint), &pDim);
    if (CL_SUCCESS != err)
    {
        LogError("Error: Failed to set argument pDim, returned %s\n", TranslateOpenCLError(err));
        return err;
    }

    err = clSetKernelArg(ocl->kernel, 5, sizeof(cl_uint), &nDim);
    if (CL_SUCCESS != err)
    {
        LogError("Error: Failed to set argument nDim, returned %s\n", TranslateOpenCLError(err));
        return err;
    }

    err = clSetKernelArg(ocl->kernel, 6, sizeof(cl_mem), (void*)matrixD);
    if (CL_SUCCESS != err)
    {
        LogError("Error: Failed to set argument matrixD, returned %s\n", TranslateOpenCLError(err));
        return err;
    }

    if (kernel == 3) {//here it's the learning rate
        err = clSetKernelArg(ocl->kernel, 7, sizeof(cl_float), &learningRate);
        if (CL_SUCCESS != err)
        {
            LogError("Error: Failed to set argument learning_rate, returned %s\n", TranslateOpenCLError(err));
            return err;
        }
    }

    return err;
}

cl_uint mExecuteMultiplyKernelCustom(ocl_args_d_t* ocl, cl_uint mDim, cl_uint nDim)
{
    cl_int err = CL_SUCCESS;

    // Define global iteration space for clEnqueueNDRangeKernel.
    const int WGS = 16;
    const int TW = 16;
    const size_t global[2] = { mDim, nDim/TW};
    const size_t local[2] = { WGS, WGS/TW};
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

cl_uint executeMultiplyKernel(ocl_args_d_t* ocl, const size_t global[2],  const size_t local[2])
{
    cl_int err = CL_SUCCESS;

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

    const size_t global[2] = { mDim, nDim};

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

cl_uint forwardpassClassifier(ocl_args_d_t* ocl, cl_mem* buffersWeightsArray, cl_mem* buffersBiasesArray, cl_mem* buffersOutsArray, cl_mem* bufferInputArray, 
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

        cl_uint optimizedSizeNetworkOutput = ((sizeof(cl_float) * mDim* nDim - 1) / 64 + 1) * 64;
        cl_int err;
        if (mDim % 16 == 0 && nDim % 16 == 0) {
            ocl->kernel = activationFunctionKernels[ActivationFunctions[x]];
            if (CL_SUCCESS != mSetKernelArguments(ocl, &(buffersBiasesArray[x]), mDim, pDim, nDim, 0.0, 1))
            {
                return -1;
            }
            if (CL_SUCCESS != mExecuteMultiplyKernelCustom(ocl,mDim ,nDim))
            {
                return -1;
            }
        }
        else {
            ocl->kernel = activationFunctionKernelsSimple[ActivationFunctions[x]];
            if (CL_SUCCESS != mSetKernelArguments(ocl, &(buffersBiasesArray[x]), mDim, pDim, nDim, 0.0, 1))
            {
                return -1;
            }
            if (CL_SUCCESS != mExecuteMultiplyKernel(ocl, mDim, nDim))
            {
                return -1;
            }
        }
    }
}



cl_uint backpropClassifier(ocl_args_d_t* ocl, cl_mem* buffersWeightsArray, cl_mem* buffersBiasesArray, cl_mem* buffersOutsArray, cl_mem* buffersDeltasArray, cl_mem* bufferInputArray, int dimensions[],
    int* ActivationFunctions, cl_kernel* activationFunctionDeltaKernels, cl_kernel* activationFunctionDeltaKernelsSimple, cl_float* correctOutput, cl_float* costs, cl_float learning_rate, int iter, int batchSize, int layers, int classes) {

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
        choices[i] = idx;
        for (int j = 0; j < classes; ++j) {
            preSoftmaxOutputs[j * batchSize + i] = preSoftmaxOutputs[j * batchSize + i] - maxval;
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
            if (CL_SUCCESS != mSetKernelArguments(ocl, &outputs, mDim, pDim, nDim, 0.0, 2))
            {
                return -1;
            }
            if (CL_SUCCESS != mExecuteMultiplyKernelCustom(ocl, mDim, nDim))
            {
                return -1;
            }
        }
        else {
            ocl->kernel = activationFunctionDeltaKernelsSimple[ActivationFunctions[x]];
            if (CL_SUCCESS != mSetKernelArguments(ocl, &outputs, mDim, pDim, nDim, 0.0, 2))
            {
                return -1;
            }
            if (CL_SUCCESS != mExecuteMultiplyKernel(ocl, mDim, nDim))
            {
                return -1;
            }
        }
    }

    //perform weight updates now. We can potentially parallelize this fully even across all network layers
    //but for now it's only across the weights of each layer and then sequentially across layers
    pDim = batchSize;
    cl_kernel optimKernel = clCreateKernel(ocl->program, "Update_Weights_Buffers", &err);
    cl_kernel simpleKernel = clCreateKernel(ocl->programSimple, "Update_Weights_Buffers", &err);
    if (CL_SUCCESS != err)
    {
        LogError("Error: clCreateKernel returned %s\n", TranslateOpenCLError(err));
        return -1;
    }
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
            ocl->kernel = optimKernel;
            if (CL_SUCCESS != mSetKernelArguments(ocl, &(buffersBiasesArray[x]), mDim, pDim, nDim, learning_rate, 3))
            {
                return -1;
            }

            if (CL_SUCCESS != mExecuteMultiplyKernelCustom(ocl, mDim, nDim)) //thankfully, the same function works for our Multiply_Deltas kernel too
            {
                return -1;
            }
        }
        else {
            ocl->kernel = simpleKernel;
            if (CL_SUCCESS != mSetKernelArguments(ocl, &(buffersBiasesArray[x]), mDim, pDim, nDim, learning_rate, 3))
            {
                return -1;
            }

            if (CL_SUCCESS != mExecuteMultiplyKernel(ocl, mDim, nDim)) //thankfully, the same function works for our Multiply_Deltas kernel too
            {
                return -1;
            }
        }
    }
    clReleaseKernel(optimKernel);
    clReleaseKernel(simpleKernel);
    _aligned_free(preSoftmaxOutputs);
    _aligned_free(softmaxOutputs);
    _aligned_free(deltas);
    _aligned_free(choices);
    clReleaseKernel(optimKernel);
    clReleaseKernel(simpleKernel);
}

cl_uint backpropClassifier2(ocl_args_d_t* ocl, cl_mem* buffersWeightsArray, cl_mem* buffersBiasesArray, cl_mem* buffersOutsArray, cl_mem* buffersDeltasArray, cl_mem* bufferInputArray, int dimensions[],
    int* ActivationFunctions, cl_kernel* activationFunctionDeltaKernels, cl_kernel* activationFunctionDeltaKernelsSimple, cl_float* correctOutput, cl_float* costs, cl_float learning_rate, int iter, int batchSize, int layers, int classes) {

    std::cout << "In backprop for iter " << iter << " \n";
    cl_int err = CL_SUCCESS;

    cl_uint optimizedSizeNetworkOutput = ((sizeof(cl_float) * batchSize * classes - 1) / 64 + 1) * 64;
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
        choices[i] = idx;
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

    err = clEnqueueWriteBuffer(ocl->commandQueue, buffersDeltasArray[layers - 1], true, 0, sizeof(cl_float) * batchSize * classes, deltas, 0, NULL, NULL);
    if (CL_SUCCESS != err)
    {
        LogError("Error: clEnqueueReadBuffer returned %s\n", TranslateOpenCLError(err));
    }

    //non-output deltas calculation loop
    ocl->dstMem = buffersDeltasArray[layers - 1];
    cl_mem outputs;
    cl_kernel optimKernel = clCreateKernel(ocl->program, "Update_Weights_Buffers", &err);
    cl_kernel simpleKernel = clCreateKernel(ocl->programSimple, "Update_Weights_Buffers", &err);
    if (CL_SUCCESS != err)
    {
        LogError("Error: clCreateKernel returned %s\n", TranslateOpenCLError(err));
        return -1;
    }
    int mDim, pDim, nDim;
    for (int x = layers - 1; x >= 0; --x) {


        mDim = dimensions[x];
        pDim = dimensions[x + 1];
        nDim = batchSize;
        ocl->srcA = buffersWeightsArray[x];
        ocl->srcB = buffersDeltasArray[x];
        ocl->dstMem = buffersDeltasArray[x - 1];
        outputs = buffersOutsArray[x - 1];
        if (x != 0) {
            if (mDim % 16 == 0 && nDim % 16 == 0) {
                ocl->kernel = activationFunctionDeltaKernels[ActivationFunctions[x]];
                if (CL_SUCCESS != mSetKernelArguments(ocl, &outputs, mDim, pDim, nDim, 0.0, 2))
                {
                    return -1;
                }
                if (CL_SUCCESS != mExecuteMultiplyKernelCustom(ocl, mDim, nDim))
                {
                    return -1;
                }
            }
            else {
                ocl->kernel = activationFunctionDeltaKernelsSimple[ActivationFunctions[x]];
                if (CL_SUCCESS != mSetKernelArguments(ocl, &outputs, mDim, pDim, nDim, 0.0, 2))
                {
                    return -1;
                }
                if (CL_SUCCESS != mExecuteMultiplyKernel(ocl, mDim, nDim))
                {
                    return -1;
                }
            }
        }

        mDim = dimensions[x + 1];
        pDim = batchSize;
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
            ocl->kernel = optimKernel;
            if (CL_SUCCESS != mSetKernelArguments(ocl, &(buffersBiasesArray[x]), mDim, pDim, nDim, learning_rate, 3))
            {
                return -1;
            }

            if (CL_SUCCESS != mExecuteMultiplyKernelCustom(ocl, mDim, nDim)) //thankfully, the same function works for our Multiply_Deltas kernel too
            {
                return -1;
            }
        }
        else {
            ocl->kernel = simpleKernel;
            if (CL_SUCCESS != mSetKernelArguments(ocl, &(buffersBiasesArray[x]), mDim, pDim, nDim, learning_rate, 3))
            {
                return -1;
            }

            if (CL_SUCCESS != mExecuteMultiplyKernel(ocl, mDim, nDim)) //thankfully, the same function works for our Multiply_Deltas kernel too
            {
                return -1;
            }
        }
    }

    //perform weight updates now. We can potentially parallelize this fully even across all network layers
    //but for now it's only across the weights of each layer and then sequentially across layers
    pDim = batchSize;
    clReleaseKernel(optimKernel);
    clReleaseKernel(simpleKernel);
    _aligned_free(preSoftmaxOutputs);
    _aligned_free(softmaxOutputs);
    _aligned_free(choices);
    _aligned_free(deltas);
    clReleaseKernel(optimKernel);
    clReleaseKernel(simpleKernel);
}

cl_uint backpropClassifier3(ocl_args_d_t* ocl, cl_mem* buffersWeightsArray, cl_mem* buffersBiasesArray, cl_mem* buffersOutsArray, cl_mem* buffersDeltasArray, cl_mem* bufferInputArray, int dimensions[],
    int* ActivationFunctions, cl_kernel* activationFunctionDeltaKernels, cl_kernel* activationFunctionDeltaKernelsSimple, cl_float* correctOutput, cl_float* costs, cl_float learning_rate, int iter, int batchSize, int layers, int classes) {

    std::cout << "In backprop for iter " << iter << " \n";
    cl_int err = CL_SUCCESS;

    cl_uint optimizedSizeNetworkOutput = ((sizeof(cl_float) * batchSize * classes - 1) / 64 + 1) * 64;
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
        choices[i] = idx;
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

    err = clEnqueueWriteBuffer(ocl->commandQueue, buffersDeltasArray[1], true, 0, sizeof(cl_float) * batchSize * classes, deltas, 0, NULL, NULL);
    if (CL_SUCCESS != err)
    {
        LogError("Error: clEnqueueReadBuffer returned %s\n", TranslateOpenCLError(err));
    }

    //non-output deltas calculation loop
    ocl->dstMem = buffersDeltasArray[1];
    ocl->srcB = buffersDeltasArray[0];
    cl_mem outputs;
    cl_kernel optimKernel = clCreateKernel(ocl->program, "Update_Weights_Buffers", &err);
    cl_kernel simpleKernel = clCreateKernel(ocl->programSimple, "Update_Weights_Buffers", &err);
    if (CL_SUCCESS != err)
    {
        LogError("Error: clCreateKernel returned %s\n", TranslateOpenCLError(err));
        return -1;
    }
    int mDim, pDim, nDim;
    cl_mem tempcl_mem1, tempcl_mem2;
    for (int x = layers - 1; x >= 0; --x) {
        
        mDim = dimensions[x];
        pDim = dimensions[x + 1];
        nDim = batchSize;
        tempcl_mem1 = ocl->srcB;
        ocl->srcA = buffersWeightsArray[x];
        ocl->srcB = ocl-> dstMem;
        ocl->dstMem = tempcl_mem1;
        outputs = buffersOutsArray[x - 1];
        if (x != 0) {
            if (mDim % 16 == 0 && nDim % 16 == 0) {
                ocl->kernel = activationFunctionDeltaKernels[ActivationFunctions[x]];
                if (CL_SUCCESS != mSetKernelArguments(ocl, &outputs, mDim, pDim, nDim, 0.0, 2))
                {
                    return -1;
                }
                if (CL_SUCCESS != mExecuteMultiplyKernelCustom(ocl, mDim, nDim))
                {
                    return -1;
                }
            }
            else {
                ocl->kernel = activationFunctionDeltaKernelsSimple[ActivationFunctions[x]];
                if (CL_SUCCESS != mSetKernelArguments(ocl, &outputs, mDim, pDim, nDim, 0.0, 2))
                {
                    return -1;
                }
                if (CL_SUCCESS != mExecuteMultiplyKernel(ocl, mDim, nDim))
                {
                    return -1;
                }
            }
        }

        mDim = dimensions[x + 1];
        pDim = batchSize;
        nDim = dimensions[x];
        tempcl_mem1 = ocl->srcB;
        tempcl_mem2 = ocl->dstMem;
        ocl->srcA = ocl->srcB;
        ocl->dstMem = buffersWeightsArray[x];
        if (x != 0) {
            ocl->srcB = buffersOutsArray[x - 1];
        }
        else {
            ocl->srcB = *bufferInputArray;
        }

        if (mDim % 16 == 0 && nDim % 16 == 0) {
            ocl->kernel = optimKernel;
            if (CL_SUCCESS != mSetKernelArguments(ocl, &(buffersBiasesArray[x]), mDim, pDim, nDim, learning_rate, 3))
            {
                return -1;
            }

            if (CL_SUCCESS != mExecuteMultiplyKernelCustom(ocl, mDim, nDim)) //thankfully, the same function works for our Multiply_Deltas kernel too
            {
                return -1;
            }
        }
        else {
            ocl->kernel = simpleKernel;
            if (CL_SUCCESS != mSetKernelArguments(ocl, &(buffersBiasesArray[x]), mDim, pDim, nDim, learning_rate, 3))
            {
                return -1;
            }

            if (CL_SUCCESS != mExecuteMultiplyKernel(ocl, mDim, nDim)) //thankfully, the same function works for our Multiply_Deltas kernel too
            {
                return -1;
            }
        }
        ocl->srcB = tempcl_mem1;
        ocl->dstMem = tempcl_mem2;
    }

    //perform weight updates now. We can potentially parallelize this fully even across all network layers
    //but for now it's only across the weights of each layer and then sequentially across layers
    pDim = batchSize;
    clReleaseKernel(optimKernel);
    clReleaseKernel(simpleKernel);
    _aligned_free(preSoftmaxOutputs);
    _aligned_free(softmaxOutputs);
    _aligned_free(deltas);
    _aligned_free(choices);
    clReleaseKernel(optimKernel);
    clReleaseKernel(simpleKernel);
}

cl_uint testingClassifier(ocl_args_d_t* ocl, cl_mem* buffersWeightsArray, cl_mem* buffersBiasesArray, cl_mem* buffersOutsArray, cl_mem* bufferInputArray, int dimensions[],
    int* ActivationFunctions, cl_kernel* activationFunctionKernels, cl_kernel* activationFunctionKernelsSimple, int layers, int classes, uchar** valDataset, uchar* valLabels, int numValImages ) {
    
    std::cout << "In validation \n";

    cl_int err = CL_SUCCESS;

    cl_uint optimizedSizeIn = ((sizeof(cl_float) * dimensions[0] * numValImages - 1) / 64 + 1) * 64;
    cl_float* inArray = (cl_float*)_aligned_malloc(optimizedSizeIn, 4096); //array of network input

    std::cout << "Allocated size is " << optimizedSizeIn << '\n';
    if (inArray == NULL) {
        std::cout << "inArray pointer is NULL" << '\n';
    }

    for (int i = 0; i < numValImages; ++i) {
        for (int j = 0; j < dimensions[0]; ++j) {
            cl_float temp = (cl_float)valDataset[i][j];
            inArray[j * numValImages + i] = temp;
        }
    }
    cl_uint optimizedSizeOut = ((sizeof(cl_float) * numValImages - 1) / 64 + 1) * 64;
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
            // Passing arguments into OpenCL kernel.
            if (CL_SUCCESS != mSetKernelArguments(ocl, &(buffersBiasesArray[x]), mDim, pDim, nDim, 0.0, 1))
            {
                return -1;
            }
            if (CL_SUCCESS != mExecuteMultiplyKernelCustom(ocl, mDim, nDim))
            {
                return -1;
            }
        }
        else {
            ocl->kernel = activationFunctionKernelsSimple[ActivationFunctions[x]];
            // Passing arguments into OpenCL kernel.
            if (CL_SUCCESS != mSetKernelArguments(ocl, &(buffersBiasesArray[x]), mDim, pDim, nDim, 0.0, 1))
            {
                return -1;
            }
            if (CL_SUCCESS != mExecuteMultiplyKernel(ocl, mDim, nDim))
            {
                return -1;
            }
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
    _aligned_free(softmaxOutputs);
    _aligned_free(preSoftmaxOutputs);
    _aligned_free(inArray);
    for (int i = 0; i < numValImages; ++i) {
        free(valDataset[i]);
    }
    free(valDataset);
    free(valLabels);
    std::cout << "Validation set accuracy is " << accuracy << '\n';
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
            matrixC[idx] = temp+ biasesArray[i];
            //std::cout << "le IDENTITY issa " << matrixC[idx] << '\n';
        }
    }
}

//Performing C = AxB and then elementwise sigmoid() on C
//x is normal matrix multiplication
//A = weights, B = outputs for layer l-1, C = outputs for l, 
//where l is the current layer visited in this iteration of the forwardpassCpp loop
void multiplySigmoidKernelCpp( float* matrixA, float* matrixB, float* matrixC,
    const int mDim, const int pDim, const int nDim, float* biasesArray) {
    int idx;
    float temp;
    for (int i = 0; i < mDim; i++) {
        for (int j = 0; j < nDim; j++) {
            idx = i * nDim + j;
            temp = 0.0f;
            for (int k = 0; k< pDim; k++) {
                temp += matrixA[i*pDim+k] * matrixB[j+k*nDim];
            }
            matrixC[idx] = (tanh((temp+ biasesArray[i])/2)+1)/2; //expressed sig as tanh because sig is not implemented in cmath
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
            matrixC[idx] = tanh(temp+biasesArray[i]);
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
            matrixC[idx] = fmax(temp+biasesArray[i],0.0f);
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
            matrixC[idx] = fmin(fmax(temp,-0.005f),0.005f);
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
            matrixC[idx] = fmin(fmax(temp * matrixD[idx] * (1.0f - matrixD[idx]),-0.005f),0.005f);
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
            matrixC[idx] = fmin(fmax(temp * (1.0f - pow(matrixD[idx], 2)),-0.005f),0.005f);
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
            matrixC[idx] = fmin(fmax(temp* (matrixD[idx] > 0.0 ? 1.0 : 0.0),-0.005f),0.005f);
        }
    }
}
/////////////////////////////////////////////////////////////////////////////////////BP4///////////////////////////////////////
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

    cl_uint optimizedSize = ((sizeof(float) * batchSize * classes - 1) / 64 + 1) * 64;
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
                deltaArrays[layers-1][j * batchSize + i] = softmaxOutputs[j * batchSize + i] - 1.0f;
            }
            else {
                deltaArrays[layers-1][j * batchSize + i] = softmaxOutputs[j * batchSize + i];
            }
        }
    }

    costs[iter] = AccuracyFunction(groundTruthArray, choices, batchSize);

    //non-output deltas calculation loop
    float* srcA, * srcB, * dstMem;
    dstMem = deltaArrays[layers-1];
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

    cl_uint optimizedSize = ((sizeof(float) * dimensions[0] * numValImages - 1) / 64 + 1) * 64;
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
    for (cl_uint x = 0; x < layers; ++x) {
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

    float **weightsAr;
    float** weightArrays, ** biasArrays, ** outputArrays, ** deltaArrays, ** inputArrays, ** groundTruthArrays;
    float* costs;

    std::cout << "Num of val images is " << numValImages << '\n';

    int itersPerEpoch = (numTrainImages - 1) / batchSize + 1;
    int iterations = epochs * (itersPerEpoch - 1);

    cl_uint optimizedSize = ((sizeof(float*) * layers - 1) / 64 + 1) * 64;
    weightArrays = (float**)_aligned_malloc(optimizedSize, 4096);//array of memory objects, where each memory object is a buffer of weights between layers
    biasArrays = (float**)_aligned_malloc(optimizedSize, 4096); //array of memory objects, where each memory object is a buffer of biases for some layer
    outputArrays = (float**)_aligned_malloc(optimizedSize, 4096); //array of memory objects, where each memory object is a buffer image of outputs of layers
    deltaArrays = (float**)_aligned_malloc(optimizedSize, 4096); //array of memory objects, where each memory object is an image of deltas of layers

    optimizedSize = ((sizeof(float*) * (itersPerEpoch - 1) - 1) / 64 + 1) * 64;
    inputArrays = (float**)_aligned_malloc(optimizedSize, 4096);
    groundTruthArrays = (float**)_aligned_malloc(optimizedSize, 4096); //array of network grount truth

    cl_uint optimizedSizeIn = ((sizeof(float) * dimensions[0] * batchSize - 1) / 64 + 1) * 64;
    cl_uint optimizedSizeOut = ((sizeof(float) * batchSize - 1) / 64 + 1) * 64;

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
    for (cl_uint x = 0; x < layers; ++x) {
        _aligned_free(outputArrays[x]);
        _aligned_free(deltaArrays[x]);
    }
    _aligned_free(deltaArrays);
    _aligned_free(costs);

    testingClassifierCpp(weightArrays, biasArrays, outputArrays, dimensions, activationFunctions, layers, classes, valDataset, valLabels, numValImages);

    for (cl_uint x = 0; x < layers; ++x) {
        //std::cout << "releasing obj num" << x;
        _aligned_free(weightArrays[x]);
        _aligned_free(biasArrays[x]);
    }
    for (cl_uint x = 0; x < itersPerEpoch - 1; ++x) {
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

        int optimizedSize = ((sizeof(uchar) * number_of_labels - 1) / 64 + 1) * 64;
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

cl_uint minibatchGD(ocl_args_d_t* ocl, int dimensions[], int* activationFunctions, cl_kernel* activationFunctionKernels, cl_kernel* activationFunctionKernelsSimple,
    cl_kernel* activationFunctionDeltaKernels, cl_kernel* activationFunctionDeltaKernelsSimple, int batchSize, int layers, int classes, int epochs,
    uchar** dataset, uchar* labels, int numTrainImages, uchar**valDataset, uchar* valLabels, int numValImages) {

    cl_float** weightsArray;
    cl_mem* weightBuffers, * biasBuffers, * outputBuffers, * deltaBuffers, * inputBuffer;//, *groundTruthBuffer;
    cl_float* costs;

    std::cout << "Num of val images is " << numValImages << '\n';

    int itersPerEpoch = (numTrainImages -1)/ batchSize + 1;
    int iterations = epochs * (itersPerEpoch-1);

    cl_int err = CL_SUCCESS;
    cl_uint optimizedSize = ((sizeof(cl_mem) * layers - 1) / 64 + 1) * 64;
    weightBuffers = (cl_mem*)_aligned_malloc(optimizedSize, 4096); //array of memory objects, where each memory object is a buffer of weights between layers
    biasBuffers = (cl_mem*)_aligned_malloc(optimizedSize, 4096); //array of memory objects, where each memory object is a buffer of biases for some layer
    outputBuffers = (cl_mem*)_aligned_malloc(optimizedSize, 4096); //array of memory objects, where each memory object is a buffer image of outputs of layers
    deltaBuffers = (cl_mem*)_aligned_malloc(optimizedSize, 4096); //array of memory objects, where each memory object is an image of deltas of layers

    optimizedSize = ((sizeof(cl_mem) * (itersPerEpoch-1) - 1) / 64 + 1) * 64;
    inputBuffer = (cl_mem*)_aligned_malloc(optimizedSize, 4096);

    cl_uint optimizedSizeIn = ((sizeof(cl_float) * dimensions[0] * batchSize - 1) / 64 + 1) * 64;
    cl_uint optimizedSizeOut = ((sizeof(cl_float*) * (itersPerEpoch-1) - 1) / 64 + 1) * 64;
    cl_float* inArray = (cl_float*)_aligned_malloc(optimizedSizeIn, 4096); //array of network input
    cl_float** correctOutput = (cl_float**)_aligned_malloc(optimizedSizeOut, 4096); //array of network grount truth

    optimizedSizeOut = ((sizeof(cl_float) * batchSize - 1) / 64 + 1) * 64;
    for (int iter = 0; iter < itersPerEpoch-1; ++iter) {
        correctOutput[iter] = (cl_float*)_aligned_malloc(optimizedSizeOut, 4096);
        for (int i = 0; i < batchSize; ++i) {
            for (int j = 0; j < dimensions[0]; ++j) {
                inArray[j * batchSize + i] = (cl_float)dataset[i + batchSize*iter][j];
            }
            correctOutput[iter][i] = labels[i + batchSize * iter];
        }
        inputBuffer[iter] = clCreateBuffer(ocl->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, optimizedSizeIn, inArray, &err);
        if (CL_SUCCESS != err)
        {
            LogError("Error: creating input buffer returned %s\n", TranslateOpenCLError(err));
            return err;
        }
    }

    for (int i = 0; i < numTrainImages; i++) {
        free(dataset[i]);
    }
    free(dataset);
    free(labels);

    if (NULL == weightBuffers || NULL == biasBuffers||NULL == outputBuffers || NULL == deltaBuffers || NULL == inputBuffer || NULL == correctOutput)
    {
        LogError("Error: _aligned_malloc failed to allocate buffers.\n");
        return -1;
    }

    //initializing weights, outputs and delta buffers
    int optimizedSizeCosts = ((sizeof(cl_float) *epochs*(itersPerEpoch-1) - 1) / 64 + 1) * 64;
    costs = (cl_float*)_aligned_malloc(optimizedSizeCosts, 4096);

    cl_uint optimizedSize1;
    cl_uint optimizedSize2;
    cl_float* tempWeightArray;
    cl_float* tempBiasArray;
    int mDim, pDim;

    for (cl_uint x = 0; x < layers; ++x) {
        mDim = dimensions[x + 1];
        pDim = dimensions[x];
        optimizedSize1 = ((sizeof(cl_float) * mDim * pDim - 1) / 64 + 1) * 64;
        optimizedSize2 = ((sizeof(cl_float) * mDim - 1) / 64 + 1) * 64;
        tempWeightArray = (cl_float*)_aligned_malloc(optimizedSize1, 4096);
        tempBiasArray = (cl_float*)_aligned_malloc(optimizedSize2, 4096);
        std::cout << "Weights of layer " << x << " are: \n";
        mGenerateMatrices(tempWeightArray, mDim, pDim);
        mGenerateMatrices(tempBiasArray, mDim, 1);

        // Create first buffer based on host memory inputA
        weightBuffers[x] = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * mDim * pDim, tempWeightArray, &err);
        if (CL_SUCCESS != err)
        {
            LogError("Error: clCreateImage for srcA returned %s\n", TranslateOpenCLError(err));
            return err;
        }

        biasBuffers[x] = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * mDim, tempBiasArray, &err);
        if (CL_SUCCESS != err)
        {
            LogError("Error: clCreateImage for srcA returned %s\n", TranslateOpenCLError(err));
            return err;
        }

        outputBuffers[x] = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE, sizeof(cl_float) * mDim * batchSize, NULL, &err);
        if (CL_SUCCESS != err)
        {
            LogError("Error: clCreateImage for srcA returned %s\n", TranslateOpenCLError(err));
            return err;
        }

        deltaBuffers[x] = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE, sizeof(cl_float) * mDim * batchSize, NULL, &err);
        if (CL_SUCCESS != err)
        {
            LogError("Error: clCreateImage for srcA returned %s\n", TranslateOpenCLError(err));
            return err;
        }
        _aligned_free(tempWeightArray);
        _aligned_free(tempBiasArray);
    }


    cl_float learning_rate = 0.0008;
    cl_float temptot;
    std::cout << "iters per epoch is " << (itersPerEpoch-1) << '\n';
    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::cout << "epoch " << epoch << " hereeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee \n";
        int batchSizeTemp = batchSize;
        learning_rate *= 0.92;
        for (int i = 0; i < itersPerEpoch-1; ++i) {
            std::cout << "ENTERING OPENCL FORWARD" << "\n";
            forwardpassClassifier(ocl, weightBuffers, biasBuffers, outputBuffers, &(inputBuffer[i]), dimensions, activationFunctions, activationFunctionKernels, activationFunctionKernelsSimple, batchSizeTemp, layers);
            std::cout << "ENTERING OPENCL BACKWARD" << "\n";
            backpropClassifier(ocl, weightBuffers, biasBuffers, outputBuffers, deltaBuffers, &(inputBuffer[i]), dimensions, activationFunctions, activationFunctionDeltaKernels, activationFunctionDeltaKernelsSimple, correctOutput[i], costs, learning_rate, (epoch*(itersPerEpoch-1)+i), batchSizeTemp, layers, classes);
        }
        temptot = 0.0f;
        for (int i = 0; i < itersPerEpoch - 1; ++i) {
            temptot += costs[epoch * (itersPerEpoch - 1) + i];
        }
        std::cout << "average accuracy for epoch " << epoch << " is " << temptot / (itersPerEpoch-1) << '\n';
    }

    for (int i = 0; i < iterations; i++) {
        std::cout << costs[i] << '\n';
    }

    std::cout<<"dimension[0] is "<<dimensions[0]<<'\n';
    std::cout << "numValImages is " << numValImages << '\n';
    //For reasonably sized validation datasets, a single pass with a wide matrix is enough
    testingClassifier(ocl, weightBuffers, biasBuffers, outputBuffers, &(inputBuffer[0]), dimensions, activationFunctions, activationFunctionKernels,
        activationFunctionKernelsSimple, layers, classes, valDataset, valLabels, numValImages);

    for (cl_uint x = 0; x < layers; ++x) {
        //std::cout << "releasing obj num" << x;
        clReleaseMemObject(outputBuffers[x]);
        clReleaseMemObject(weightBuffers[x]);
        clReleaseMemObject(biasBuffers[x]);
        clReleaseMemObject(deltaBuffers[x]);
    }
    for (cl_uint x = 0; x < itersPerEpoch-1; ++x) {
        clReleaseMemObject(inputBuffer[x]);
        _aligned_free(correctOutput[x]);
    }

    _aligned_free(correctOutput);
    _aligned_free(weightBuffers);
    _aligned_free(biasBuffers);
    _aligned_free(outputBuffers);
    _aligned_free(deltaBuffers);
    _aligned_free(inputBuffer);
    _aligned_free(costs);
}

cl_uint minibatchGD2(ocl_args_d_t* ocl, int dimensions[], int* activationFunctions, cl_kernel* activationFunctionKernels, cl_kernel* activationFunctionKernelsSimple,
    cl_kernel* activationFunctionDeltaKernels, cl_kernel* activationFunctionDeltaKernelsSimple, int batchSize, int layers, int classes, int epochs,
    uchar** dataset, uchar* labels, int numTrainImages, uchar** valDataset, uchar* valLabels, int numValImages) {

    cl_float** weightsArray;
    cl_mem* weightBuffers, * biasBuffers, * outputBuffers, * deltaBuffers, * inputBuffer;//, *groundTruthBuffer;
    cl_float* costs;

    std::cout << "Num of val images is " << numValImages << '\n';

    int itersPerEpoch = (numTrainImages - 1) / batchSize + 1;
    int iterations = epochs * (itersPerEpoch - 1);

    cl_int err = CL_SUCCESS;
    cl_uint optimizedSize = ((sizeof(cl_mem) * layers - 1) / 64 + 1) * 64;
    weightBuffers = (cl_mem*)_aligned_malloc(optimizedSize, 4096); //array of memory objects, where each memory object is a buffer of weights between layers
    biasBuffers = (cl_mem*)_aligned_malloc(optimizedSize, 4096); //array of memory objects, where each memory object is a buffer of biases for some layer
    outputBuffers = (cl_mem*)_aligned_malloc(optimizedSize, 4096); //array of memory objects, where each memory object is a buffer image of outputs of layers
    deltaBuffers = (cl_mem*)_aligned_malloc(optimizedSize, 4096); //array of memory objects, where each memory object is an image of deltas of layers

    optimizedSize = ((sizeof(cl_mem) * (itersPerEpoch - 1) - 1) / 64 + 1) * 64;
    inputBuffer = (cl_mem*)_aligned_malloc(optimizedSize, 4096);

    cl_uint optimizedSizeIn = ((sizeof(cl_float) * dimensions[0] * batchSize - 1) / 64 + 1) * 64;
    cl_uint optimizedSizeOut = ((sizeof(cl_float*) * (itersPerEpoch - 1) - 1) / 64 + 1) * 64;
    cl_float* inArray = (cl_float*)_aligned_malloc(optimizedSizeIn, 4096); //array of network input
    cl_float** correctOutput = (cl_float**)_aligned_malloc(optimizedSizeOut, 4096); //array of network grount truth

    optimizedSizeOut = ((sizeof(cl_float) * batchSize - 1) / 64 + 1) * 64;
    for (int iter = 0; iter < itersPerEpoch - 1; ++iter) {
        correctOutput[iter] = (cl_float*)_aligned_malloc(optimizedSizeOut, 4096);
        for (int i = 0; i < batchSize; ++i) {
            for (int j = 0; j < dimensions[0]; ++j) {
                inArray[j * batchSize + i] = (cl_float)dataset[i + batchSize * iter][j];
            }
            correctOutput[iter][i] = labels[i + batchSize * iter];
        }
        inputBuffer[iter] = clCreateBuffer(ocl->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, optimizedSizeIn, inArray, &err);
        if (CL_SUCCESS != err)
        {
            LogError("Error: creating input buffer returned %s\n", TranslateOpenCLError(err));
            return err;
        }
    }

    for (int i = 0; i < numTrainImages; i++) {
        free(dataset[i]);
    }
    free(dataset);
    free(labels);

    if (NULL == weightBuffers || NULL == biasBuffers || NULL == outputBuffers || NULL == deltaBuffers || NULL == inputBuffer || NULL == correctOutput)
    {
        LogError("Error: _aligned_malloc failed to allocate buffers.\n");
        return -1;
    }

    //initializing weights, outputs and delta buffers
    int optimizedSizeCosts = ((sizeof(cl_float) * epochs * (itersPerEpoch - 1) - 1) / 64 + 1) * 64;
    costs = (cl_float*)_aligned_malloc(optimizedSizeCosts, 4096);

    cl_uint optimizedSize1;
    cl_uint optimizedSize2;
    cl_float* tempWeightArray;
    cl_float* tempBiasArray;
    int mDim, pDim;

    for (cl_uint x = 0; x < layers; ++x) {
        mDim = dimensions[x + 1];
        pDim = dimensions[x];
        optimizedSize1 = ((sizeof(cl_float) * mDim * pDim - 1) / 64 + 1) * 64;
        optimizedSize2 = ((sizeof(cl_float) * mDim - 1) / 64 + 1) * 64;
        tempWeightArray = (cl_float*)_aligned_malloc(optimizedSize1, 4096);
        tempBiasArray = (cl_float*)_aligned_malloc(optimizedSize2, 4096);
        std::cout << "Weights of layer " << x << " are: \n";
        mGenerateMatrices(tempWeightArray, mDim, pDim);
        mGenerateMatrices(tempBiasArray, mDim, 1);

        // Create first buffer based on host memory inputA
        weightBuffers[x] = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * mDim * pDim, tempWeightArray, &err);
        if (CL_SUCCESS != err)
        {
            LogError("Error: clCreateBuffer for weightBuffers returned %s\n", TranslateOpenCLError(err));
            return err;
        }

        biasBuffers[x] = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * mDim, tempBiasArray, &err);
        if (CL_SUCCESS != err)
        {
            LogError("Error: clCreateBuffer for biasBuffers returned %s\n", TranslateOpenCLError(err));
            return err;
        }

        outputBuffers[x] = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE, sizeof(cl_float) * mDim * batchSize, NULL, &err);
        if (CL_SUCCESS != err)
        {
            LogError("Error: clCreateBuffer for outputBuffers returned %s\n", TranslateOpenCLError(err));
            return err;
        }

        _aligned_free(tempWeightArray);
        _aligned_free(tempBiasArray);
    }

    int maxmDim = 0;
    for (int i = 1; i < layers + 1; ++i) {
        if (dimensions[i] > maxmDim) {
            maxmDim = dimensions[i];
        }
    }
    deltaBuffers[0] = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE, sizeof(cl_float) * maxmDim * batchSize, NULL, &err);
    deltaBuffers[1] = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE, sizeof(cl_float) * maxmDim * batchSize, NULL, &err);
    if (CL_SUCCESS != err)
    {
        LogError("Error: clCreateBuffer for deltaBuffersreturned %s\n", TranslateOpenCLError(err));
        return err;
    }


    cl_float learning_rate = 0.0008;
    cl_float temptot;
    std::cout << "iters per epoch is " << (itersPerEpoch - 1) << '\n';
    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::cout << "epoch " << epoch << " hereeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee \n";
        int batchSizeTemp = batchSize;
        learning_rate *= 0.92;
        for (int i = 0; i < itersPerEpoch - 1; ++i) {
            std::cout << "ENTERING OPENCL FORWARD" << "\n";
            forwardpassClassifier(ocl, weightBuffers, biasBuffers, outputBuffers, &(inputBuffer[i]), dimensions, activationFunctions, activationFunctionKernels, activationFunctionKernelsSimple, batchSizeTemp, layers);
            std::cout << "ENTERING OPENCL BACKWARD" << "\n";
            backpropClassifier3(ocl, weightBuffers, biasBuffers, outputBuffers, deltaBuffers, &(inputBuffer[i]), dimensions, activationFunctions, activationFunctionDeltaKernels, activationFunctionDeltaKernelsSimple, correctOutput[i], costs, learning_rate, (epoch * (itersPerEpoch - 1) + i), batchSizeTemp, layers, classes);
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

    std::cout << "dimension[0] is " << dimensions[0] << '\n';
    std::cout << "numValImages is " << numValImages << '\n';
    //For reasonably sized validation datasets, a single pass with a wide matrix is enough
    testingClassifier(ocl, weightBuffers, biasBuffers, outputBuffers, &(inputBuffer[0]), dimensions, activationFunctions, activationFunctionKernels,
        activationFunctionKernelsSimple, layers, classes, valDataset, valLabels, numValImages);

    for (cl_uint x = 0; x < layers; ++x) {
        //std::cout << "releasing obj num" << x;
        clReleaseMemObject(outputBuffers[x]);
        clReleaseMemObject(weightBuffers[x]);
        clReleaseMemObject(biasBuffers[x]);
    }

    clReleaseMemObject(deltaBuffers[0]);
    clReleaseMemObject(deltaBuffers[1]);

    for (cl_uint x = 0; x < itersPerEpoch - 1; ++x) {
        clReleaseMemObject(inputBuffer[x]);
        _aligned_free(correctOutput[x]);
    }
    _aligned_free(weightBuffers);
    _aligned_free(biasBuffers);
    _aligned_free(outputBuffers);
    _aligned_free(deltaBuffers);
    _aligned_free(inputBuffer);
    _aligned_free(costs);
}

//Testing correctness and measuring latency of new kernels
cl_int kernelCorrectnessTesting(ocl_args_d_t* ocl, char** activationFunctionKernelNames,char** activationFunctionDeltaKernelNames, cl_kernel* activationFunctionKernels,
    cl_kernel* activationFunctionDeltaKernels, cl_kernel* activationFunctionKernelsSimple, cl_kernel* activationFunctionDeltaKernelsSimple, 
    const size_t global[2], const size_t local[2], int numActivationFunctions, int mDim, int pDim, int nDim) {
    //Testing matrices where each dimension has a different value
    cl_int err;
    bool outputIsCorrect;
    const size_t globalSimple[2] = { mDim, nDim };
 
    cl_uint optimizedSizeTempA = ((sizeof(cl_float) * mDim * pDim - 1) / 64 + 1) * 64;
    cl_uint optimizedSizeTempB = ((sizeof(cl_float) * pDim * nDim - 1) / 64 + 1) * 64;
    cl_uint optimizedSizeTempBiases = ((sizeof(cl_float) * mDim - 1) / 64 + 1) * 64;
    cl_uint optimizedSizeTempC = ((sizeof(cl_float) * mDim * nDim - 1) / 64 + 1) * 64;
    cl_float* matrixAar = (cl_float*)_aligned_malloc(optimizedSizeTempA, 4096);
    cl_float* matrixBar = (cl_float*)_aligned_malloc(optimizedSizeTempB, 4096);
    cl_float* biasesar = (cl_float*)_aligned_malloc(optimizedSizeTempBiases, 4096);
    cl_float* auxiliaryMatrixar = (cl_float*)_aligned_malloc(optimizedSizeTempC, 4096);
    mGenerateMatrices(matrixAar, mDim, pDim);
    mGenerateMatrices(matrixBar, pDim, nDim);
    mGenerateMatrices(auxiliaryMatrixar, mDim, nDim);
    mGenerateMatrices(biasesar, mDim, 1);


    // Create first buffer based on host memory inputA
    cl_mem matrixA = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * mDim * pDim, matrixAar, &err);
    cl_mem matrixB = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * pDim * nDim, matrixBar, &err);
    cl_mem biases = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * mDim, biasesar, &err);
    cl_mem matrixC = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE, sizeof(cl_float) * mDim * nDim, NULL, &err);
    cl_mem matrixD = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE, sizeof(cl_float) * mDim * nDim, NULL, &err);
    cl_mem auxiliaryMatrix = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * mDim*nDim, auxiliaryMatrixar, &err);
    if (CL_SUCCESS != err)
    {
        LogError("Error: matrix creation failed with %s\n", TranslateOpenCLError(err));
        return err;
    }

    _aligned_free(matrixAar);
    _aligned_free(matrixBar);
    _aligned_free(biasesar);
    _aligned_free(auxiliaryMatrixar);

    cl_uint optimizedSizeNetworkOutput = ((sizeof(cl_float) * mDim * nDim - 1) / 64 + 1) * 64;
    cl_float* outputs = (cl_float*)_aligned_malloc(optimizedSizeNetworkOutput, 4096);
    cl_float* simpleOutputs = (cl_float*)_aligned_malloc(optimizedSizeNetworkOutput, 4096);
    int arraySize = mDim * nDim;

    ocl->srcA = matrixA;
    ocl->srcB = matrixB;
    auto start = high_resolution_clock::now();
    auto elapsed = std::chrono::high_resolution_clock::now() - start;
    long long microseconds;

    //FORWARDPROP KERNELS//////////////////////////////////////////////////////////////////////////////////////////////////
    for (int kernelIdx = 0; kernelIdx < numActivationFunctions; ++kernelIdx) {
        outputIsCorrect = true;
        // NEW KERNEL -----------------------------------------------------------------------------------------
        ocl->kernel = activationFunctionKernels[kernelIdx];
        ocl->dstMem = matrixC;
        if (CL_SUCCESS != mSetKernelArguments(ocl, &biases, mDim, pDim, nDim, 0.0, 1))
        {
            return -1;
        }

        start = high_resolution_clock::now();
        if (CL_SUCCESS != executeMultiplyKernel(ocl, global, local))
        {
            return -1;
        }
        elapsed = std::chrono::high_resolution_clock::now() - start;
        microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
        std::cout << microseconds << " microseconds for new "<< activationFunctionKernelNames[kernelIdx]<<"\n";

        err = clEnqueueReadBuffer(ocl->commandQueue, ocl->dstMem, true, 0, sizeof(cl_float) * arraySize, outputs, 0, NULL, NULL);
        if (CL_SUCCESS != err)
        {
            LogError("Error: clEnqueueReadBuffer returned %s\n", TranslateOpenCLError(err));
        }

        //TESTED KERNEL-------------------------------------------------------------------------------------------
        ocl->kernel = activationFunctionKernelsSimple[kernelIdx];
        ocl->dstMem = matrixD;
        // Passing arguments into OpenCL kernel.
        if (CL_SUCCESS != mSetKernelArguments(ocl, &biases, mDim, pDim, nDim, 0.0, 1))
        {
            return -1;
        }
        start = high_resolution_clock::now();
        if (CL_SUCCESS != executeMultiplyKernel(ocl, globalSimple, NULL))
        {
            return -1;
        }
        elapsed = std::chrono::high_resolution_clock::now() - start;
        microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
        std::cout << microseconds << " microseconds for tested " << activationFunctionKernelNames[kernelIdx] << "\n";

        err = clEnqueueReadBuffer(ocl->commandQueue, ocl->dstMem, true, 0, sizeof(cl_float) * arraySize, simpleOutputs, 0, NULL, NULL);
        if (CL_SUCCESS != err)
        {
            LogError("Error: clEnqueueReadBuffer returned %s\n", TranslateOpenCLError(err));
        }

        for (int i = 0; i < arraySize; ++i) {
            if (outputs[i] != simpleOutputs[i]) {
                outputIsCorrect = false;
                //std::cout << "2 VALUES ARE " << outputs[i] << " and " << simpleOutputs[i] << '\n';
            }
        }
           
        if (outputIsCorrect) {
            std::cout << "Kernel " << activationFunctionKernelNames[kernelIdx] << " PASSES" << '\n';
        }
        else {
            std::cout << "Kernel " << activationFunctionKernelNames[kernelIdx] << " FAILS" << '\n';
        }
    }

    //BACKPROP KERNELS///////////////////////////////////////////////////////////////////////////////////////////////
    for (int kernelIdx = 0; kernelIdx < numActivationFunctions; ++kernelIdx) {
        outputIsCorrect = true;
        // NEW KERNEL -----------------------------------------------------------------------------------------
        ocl->kernel = activationFunctionDeltaKernels[kernelIdx];
        ocl->dstMem = matrixC;
        if (CL_SUCCESS != mSetKernelArguments(ocl, &auxiliaryMatrix, mDim, pDim, nDim, 0.0, 2))
        {
            return -1;
        }
        start = high_resolution_clock::now();
        if (CL_SUCCESS != executeMultiplyKernel(ocl, global, local))
        {
            return -1;
        }
        elapsed = std::chrono::high_resolution_clock::now() - start;
        microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
        std::cout << microseconds << " microseconds for new " << activationFunctionDeltaKernelNames[kernelIdx] << "\n";

        err = clEnqueueReadBuffer(ocl->commandQueue, ocl->dstMem, true, 0, sizeof(cl_float) * arraySize, outputs, 0, NULL, NULL);
        if (CL_SUCCESS != err)
        {
            LogError("Error: clEnqueueReadBuffer returned %s\n", TranslateOpenCLError(err));
        }

        //TESTED KERNEL-------------------------------------------------------------------------------------------
        ocl->kernel = activationFunctionDeltaKernelsSimple[kernelIdx];
        ocl->dstMem = matrixD;
        // Passing arguments into OpenCL kernel.
        if (CL_SUCCESS != mSetKernelArguments(ocl, &auxiliaryMatrix, mDim, pDim, nDim, 0.0, 2))
        {
            return -1;
        }
        start = high_resolution_clock::now();
        if (CL_SUCCESS != executeMultiplyKernel(ocl, globalSimple, NULL))
        {
            return -1;
        }
        elapsed = std::chrono::high_resolution_clock::now() - start;
        microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
        std::cout << microseconds << " microseconds for tested " << activationFunctionDeltaKernelNames[kernelIdx] << "\n";

        err = clEnqueueReadBuffer(ocl->commandQueue, ocl->dstMem, true, 0, sizeof(cl_float) * arraySize, simpleOutputs, 0, NULL, NULL);
        if (CL_SUCCESS != err)
        {
            LogError("Error: clEnqueueReadBuffer returned %s\n", TranslateOpenCLError(err));
        }

        for (int i = 0; i < arraySize; ++i) {
            if (outputs[i] != simpleOutputs[i]) {
                outputIsCorrect = false;
            }
        }

        if (outputIsCorrect) {
            std::cout << "Kernel " << activationFunctionDeltaKernelNames[kernelIdx] << " PASSES" << '\n';
        }
        else {
            std::cout << "Kernel " << activationFunctionDeltaKernelNames[kernelIdx] << " FAILS" << '\n';
        }
    }

    //weight update kernels
     //NEW KERNEL-------------------------------------------------------------------------------------------
    cl_float learning_rate = 0.1f;
    ocl->kernel = clCreateKernel(ocl->program, "Update_Weights_Buffers", &err);
    ocl->dstMem = matrixC;
    if (CL_SUCCESS != err)
    {
        LogError("Error: clCreateKernel returned %s\n", TranslateOpenCLError(err));
        return -1;
    }
    if (CL_SUCCESS != mSetKernelArguments(ocl, &biases, mDim, pDim, nDim, learning_rate, 3))
    {
        return -1;
    }
    start = high_resolution_clock::now();
    if (CL_SUCCESS != executeMultiplyKernel(ocl, global, local))
    {
        return -1;
    }
    elapsed = std::chrono::high_resolution_clock::now() - start;
    microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    std::cout << microseconds << " microseconds for new Update_Weights_Buffers \n";

    //TESTED KERNEL-------------------------------------------------------------------------------------------
    ocl->kernel = clCreateKernel(ocl->programSimple, "Update_Weights_Buffers", &err);
    ocl->dstMem = matrixD;
    if (CL_SUCCESS != err)
    {
        LogError("Error: clCreateKernel returned %s\n", TranslateOpenCLError(err));
        return -1;
    }
    if (CL_SUCCESS != mSetKernelArguments(ocl, &biases, mDim, pDim, nDim, learning_rate, 3))
    {
        return -1;
    }
    start = high_resolution_clock::now();
    if (CL_SUCCESS != executeMultiplyKernel(ocl, globalSimple, NULL))
    {
        return -1;
    }
    elapsed = std::chrono::high_resolution_clock::now() - start;
    microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    std::cout << microseconds << " microseconds for tested Update_Weights_Buffers \n";

    for (int i = 0; i < arraySize; ++i) {
        if (outputs[i] != simpleOutputs[i]) {
            outputIsCorrect = false;
        }
    }

    if (outputIsCorrect) {
        std::cout << "Kernel Update_Weights_Buffers PASSES" << '\n';
    }
    else {
        std::cout << "Kernel Update_Weights_Buffers FAILS" << '\n';
    }

    system("pause");
    return 0;
}

//This function takes multiple measurements of latency and returns average for a specific set of dimensions
long long kernelLatencyTestingAuxiliary(ocl_args_d_t* ocl, cl_kernel* activationFunctionKernels, 
    const size_t global[2], const size_t local[2], int mDim, int pDim, int nDim, int iterations, int typeOfKernel) {
    //Testing matrices where each dimension has a different value
    cl_int err;

    cl_uint optimizedSizeTempA = ((sizeof(cl_float) * mDim * pDim - 1) / 64 + 1) * 64;
    cl_uint optimizedSizeTempB = ((sizeof(cl_float) * pDim * nDim - 1) / 64 + 1) * 64;
    cl_uint optimizedSizeTempC = ((sizeof(cl_float) * mDim * nDim - 1) / 64 + 1) * 64;
    cl_uint optimizedSizeTempBiases = ((sizeof(cl_float) * mDim - 1) / 64 + 1) * 64;
    cl_uint optimizedSizeTempOutputs = ((sizeof(cl_float) * mDim * nDim - 1) / 64 + 1) * 64;
    cl_float* matrixAar = (cl_float*)_aligned_malloc(optimizedSizeTempA, 4096);
    cl_float* matrixBar = (cl_float*)_aligned_malloc(optimizedSizeTempB, 4096);
    cl_float* matrixBiasesar = (cl_float*)_aligned_malloc(optimizedSizeTempBiases, 4096);
    cl_float* matrixOutputsar = (cl_float*)_aligned_malloc(optimizedSizeTempOutputs, 4096);
    mGenerateMatrices(matrixAar, mDim, pDim);
    mGenerateMatrices(matrixBar, pDim, nDim);
    mGenerateMatrices(matrixBiasesar, mDim, 1);
    mGenerateMatrices(matrixOutputsar, mDim, nDim);

    cl_mem matrixA = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * mDim * pDim, matrixAar, &err);
    cl_mem matrixB = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * pDim * nDim, matrixBar, &err);
    cl_mem matrixC = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE, sizeof(cl_float) * mDim * nDim, NULL, &err);
    cl_mem matrixBiases = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * mDim, matrixBiasesar, &err);
    cl_mem matrixOutputs = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * mDim * nDim, matrixOutputsar, &err);
    if (CL_SUCCESS != err)
    {
        LogError("Error: matrix creation failed with %s\n", TranslateOpenCLError(err));
        return err;
    }

    _aligned_free(matrixAar);
    _aligned_free(matrixBar);
    _aligned_free(matrixBiasesar);
    _aligned_free(matrixOutputsar);
    int arraySize = mDim * nDim;

    ocl->srcA = matrixA;
    ocl->srcB = matrixB;
    auto start = high_resolution_clock::now();
    auto elapsed = std::chrono::high_resolution_clock::now() - start;
    long long microseconds;

    cl_uint timesSize = ((sizeof(long long) * iterations - 1) / 64 + 1) * 64;
    long long* times = (long long*)_aligned_malloc(timesSize, 4096);
    for (int iter = 0; iter < iterations; ++iter) {

        ocl->kernel = *activationFunctionKernels;
        ocl->dstMem = matrixC;
        if (typeOfKernel == 1) {
            if (CL_SUCCESS != mSetKernelArguments(ocl, &matrixBiases, mDim, pDim, nDim, 0.0, 1))
            {
                return -1;
            }
        }
        else if (typeOfKernel == 2) {
            if (CL_SUCCESS != mSetKernelArguments(ocl, &matrixOutputs, mDim, pDim, nDim, 0.0, 2))
            {
                return -1;
            }
        }
        else {
            if (CL_SUCCESS != mSetKernelArguments(ocl, &matrixBiases, mDim, pDim, nDim, 0.1f, 3))
            {
                return -1;
            }
        }

        start = high_resolution_clock::now();
        if (CL_SUCCESS != executeMultiplyKernel(ocl, global, local))
        {
            return -1;
        }
        elapsed = std::chrono::high_resolution_clock::now() - start;
        microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
        //std::cout << microseconds << " microseconds for this kernel \n";
        times[iter] = microseconds;
    }


    long long avg=0;
    for (int i = 0; i < iterations; ++i) {
        avg += times[i];
    }
    avg = avg / iterations;

    clReleaseMemObject(matrixA);
    clReleaseMemObject(matrixB);
    clReleaseMemObject(matrixC);
    clReleaseMemObject(matrixBiases);
    clReleaseMemObject(matrixOutputs);
    _aligned_free(times);

    std::cout << "Average time is " << avg << '\n';
    return avg;
}

//In contrast to kernelCorrectnessTesting, this function takes multiple measurements of latency and returns average over many different sets of dimensions
void kernelLatencyTesting(ocl_args_d_t* ocl, cl_kernel* activationFunctionKernel,int iterations, int WGS, int TW, int typeOfKernel) {

    int values[1] = { 1024 };//{128, 256, 512, 1024 };
    const size_t local[2] = { WGS, WGS / TW };
    int mDim, nDim;


    long long out = 0;
    long long avg = 0;
    for (int idx = 0; idx < 1; ++idx) {
        std::cout << "in iter of idx" << '\n';
        mDim = values[idx];
        nDim = values[idx];
        const size_t global[2] = { mDim, nDim / TW };
        std::cout << "measuring for matrices of dimension " << mDim << '\n';
        if (WGS == 0) { //No work group size defined
            for (int iter = 0; iter < iterations; ++iter) {
                out = kernelLatencyTestingAuxiliary(ocl, activationFunctionKernel, global, NULL, mDim, values[idx], nDim, 1, typeOfKernel);
                avg += out;
            }
        }
        else {
            for (int iter = 0; iter < iterations; ++iter) {
                out = kernelLatencyTestingAuxiliary(ocl, activationFunctionKernel, global, local, mDim, values[idx], nDim, 1, typeOfKernel);
                avg += out;
            }
        }
    }
    avg = avg / iterations;
    std::cout << "TOTAL AVERAGE IS " << avg << '\n';
    system("pause");
    return;
}



//int _tmain(int argc, TCHAR* argv[])
//{
//    cl_int err;
//    ocl_args_d_t ocl;
//    cl_device_type deviceType = CL_DEVICE_TYPE_GPU;
//
//    cl_uint mDim = 1;
//    cl_uint pDim;
//    cl_uint nDim;
//    //A FEW EXPLANATIONS FOR HOW THIS WORKS
//    //first, we have our outer loop that performs for a set number of iterations
//    //in each iterations, we perform 3 different loops
//    //the 1st loop, corresponds to the inference phase (forward pass).
//    //The 2nd loop (and the little bit of code before it) calculates the deltas, i.e. the
//    //partial derivatives of the cost function wrt each node's pre-activation-function output.
//    //This is necessary because the way we calculate the partial derivatives of the weights is using
//    //the chain rule, and one of the 2 terms that comes up is this one (the next one is calculated
//    //in the next loop).
//    //The 3rd loop calculates the second term of the partial derivatives of the weights and updates
//    //the weights. The reason I created both a delta calculation loop and a weight update loop,
//    //is that even though the delta calculation has to happen layer by layer (since previous layer deltas
//    //are used to calculate next layer ones), the weight updates can be completely parallelized across
//    // both nodes (as with deltas) and layers. This means I can just replace the 3rd loop with one
//    //clEnqueueNDRangeKernel invocation (haven't done this yet, but will in the future).
//
//    const int batchSize = 1024;
//    const int layers = 3; //We don't count input as a layer
//    int dimAr[layers + 1] = { 784, 1024, 512, 10 }; //last layer should always be set to 1 for regression
//    //cl_float correctOutput[batchSize] = { 1.0,2.0,3.0,4.0,5.0,6.0,7.0,-77.0,-8.0,8.5,9.2,-10.0}; //The desired output
//    const int numAF = 4; //num of activation functions
//    char* activationFunctionKernelNames[numAF] = { "Multiply_Buffer_Identity",
//    "Multiply_Buffer_Sigmoid","Multiply_Buffer_Tanh","Multiply_Buffer_ReLU" };
//    char* activationFunctionDeltaKernelNames[numAF] = { "Multiply_Deltas_Buffers_Identity",
//    "Multiply_Deltas_Buffers_Sigmoid","Multiply_Deltas_Buffers_Tanh","Multiply_Deltas_Buffers_ReLU" };
//    cl_kernel activationFunctionKernels[numAF], activationFunctionDeltaKernels[numAF];
//    cl_kernel activationFunctionKernelsSimple[numAF], activationFunctionDeltaKernelsSimple[numAF];
//
//    int activationFunctions[layers] = { 3,3,0 }; // 0 for identity, 1 for tanh, 2 for sigmoid, 3 for ReLU
//
//    if (CL_SUCCESS != SetupOpenCL(&ocl, deviceType))
//    {
//        return -1;
//    }
//
//    if (CL_SUCCESS != CreateAndBuildProgram(&ocl))
//    {
//        return -1;
//    }
//
//
//    for (int x = 0; x < numAF; ++x) {
//        activationFunctionKernels[x] = clCreateKernel(ocl.program, activationFunctionKernelNames[x], &err);
//        if (CL_SUCCESS != err)
//        {
//            LogError("Error: clCreateKernel returned %s\n", TranslateOpenCLError(err));
//            return -1;
//        }
//        activationFunctionDeltaKernels[x] = clCreateKernel(ocl.program, activationFunctionDeltaKernelNames[x], &err);
//        if (CL_SUCCESS != err)
//        {
//            LogError("Error: clCreateKernel returned %s\n", TranslateOpenCLError(err));
//            return -1;
//        }
//        activationFunctionKernelsSimple[x] = clCreateKernel(ocl.programSimple, activationFunctionKernelNames[x], &err);
//        if (CL_SUCCESS != err)
//        {
//            LogError("Error: clCreateKernel returned %s\n", TranslateOpenCLError(err));
//            return -1;
//        }
//        activationFunctionDeltaKernelsSimple[x] = clCreateKernel(ocl.programSimple, activationFunctionDeltaKernelNames[x], &err);
//        if (CL_SUCCESS != err)
//        {
//            LogError("Error: clCreateKernel returned %s\n", TranslateOpenCLError(err));
//            return -1;
//        }
//    }
//
//    //Loading the dataset
//    int trainImageSize, valImageSize, numTrainImages, numTrainLabels, numValImages, numValLabels;
//    uchar** trainingDataset = read_mnist_images("C:\\Users\\george cabon x1\\source\\repos\\GPUOpenCLProjectforWindows2\\GPUOpenCLProjectforWindows2\\MNIST\\trainingImages.idx3-ubyte", numTrainImages, trainImageSize);
//    uchar* trainingLabels = read_mnist_labels("C:\\Users\\george cabon x1\\source\\repos\\GPUOpenCLProjectforWindows2\\GPUOpenCLProjectforWindows2\\MNIST\\trainingLabels.idx1-ubyte", numTrainLabels);
//
//    uchar** valDataset = read_mnist_images("C:\\Users\\george cabon x1\\source\\repos\\GPUOpenCLProjectforWindows2\\GPUOpenCLProjectforWindows2\\MNIST\\testingImages.idx3-ubyte", numValImages, valImageSize);
//    uchar* valLabels = read_mnist_labels("C:\\Users\\george cabon x1\\source\\repos\\GPUOpenCLProjectforWindows2\\GPUOpenCLProjectforWindows2\\MNIST\\testinglabels.idx1-ubyte", numValLabels);
//
//    std::cout << "There are " << numTrainImages << " with size " << valImageSize << '\n';
//    std::cout << "First image label is " << (cl_float)trainingLabels[0] << '\n';
//
//    int epochs = 1;
//    int classes = 10;

    //minibatchGDCpp(dimAr, activationFunctions, batchSize, layers, classes, epochs, trainingDataset, trainingLabels,
    //    numTrainImages, valDataset, valLabels, numValImages);

    //MAIN LOOP
    //{
    //    int WGS = 16;
    //    mDim = 1024, pDim = 768, nDim = 512;
    //    int TW = 16;
    //    const size_t global[2] = { mDim, nDim/TW };
    //    const size_t local[2] = { WGS, WGS/TW};
    //    kernelCorrectnessTesting(&ocl, activationFunctionKernelNames, activationFunctionDeltaKernelNames, activationFunctionKernels,
    //        activationFunctionDeltaKernels, activationFunctionKernelsSimple, activationFunctionDeltaKernelsSimple, global,
    //        local, numAF, mDim, pDim, nDim);
    //}

    //{
    //    int TW = 16;
    //    int WGS = 16; //set to 0 to disable manual local size setting
    //    int iterations = 1;
    //    int typeOfKernel = 1;
    //    cl_kernel weightKernel[1];
    //    weightKernel[0] = clCreateKernel(ocl.program, "Update_Weights_Buffers", &err);
    //    if (CL_SUCCESS != err)
    //    {
    //        LogError("Error: clCreateKernel returned %s\n", TranslateOpenCLError(err));
    //        return -1;
    //    }
    //    kernelLatencyTesting(&ocl, &activationFunctionKernels[0], iterations, WGS, TW, typeOfKernel);
    //    kernelLatencyTesting(&ocl, &activationFunctionDeltaKernels[0], iterations, WGS, TW, typeOfKernel);
    //    kernelLatencyTesting(&ocl, weightKernel , iterations, WGS, TW, typeOfKernel);
    //}

    //auto start = high_resolution_clock::now();
    //auto elapsed = std::chrono::high_resolution_clock::now() - start;
    //long long microseconds;

    //start = high_resolution_clock::now();
    //minibatchGD2(&ocl, dimAr, activationFunctions, activationFunctionKernels, activationFunctionKernelsSimple,
    //    activationFunctionDeltaKernels, activationFunctionDeltaKernelsSimple, batchSize, layers, classes, epochs,
    //    trainingDataset, trainingLabels, numTrainImages, valDataset, valLabels, numValImages);
    //elapsed = std::chrono::high_resolution_clock::now() - start;
    //microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    //std::cout << microseconds << " microseconds for training \n";
    //system("pause");
//}