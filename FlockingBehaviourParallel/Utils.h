#include <vector>
#include <math.h>
#include <chrono>
#include <sys/time.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <stdio.h>

#ifndef UTILS_H
#define UTILS_H

typedef unsigned long ulong;
typedef unsigned int uint;

// returns the distance between the passed vectors
float vector3Distance(const float*, const float*);
float vector3Distance(const float*, const float*, const float*, const float*, const float*, const float*);

// returns the sum of the passed vectors
void vector3Sum(const float*, const float*, float*);
void vector3Sum(const float*, const float*, const float*, const float*, const float*, const float*, float*);

// returns the subtraction of the passed vectors
void vector3Sub(const float*, const float*, float*);
void vector3Sub(const float*, const float*, const float*, const float*, const float*, const float*, float*);

// returns the multiplication of the passed vectors
void vector3Mul(const float*, const float, float*);
void vector3Mul(const float*, const float*, const float*, const float, float*);

// returns the blending of the passed vectors representing directions
void blendDirections(const float*, const float*, const float*, float*);

// returns the magnitude of the passed vector
float vector3Magnitude(const float*);
float vector3Magnitude(const float*, const float*, const float*);

// normalizes the passed vector
void vector3Normalize(float*);
void vector3Normalize(float*, float*, float*);

#define CHECK(call)                                                           \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
    }                                                                          \
}

#define CHECK_CUBLAS(call)                                                     \
{                                                                              \
    cublasStatus_t err;                                                        \
    if ((err = (call)) != CUBLAS_STATUS_SUCCESS)                               \
    {                                                                          \
        fprintf(stderr, "Got CUBLAS error %d at %s:%d\n", err, __FILE__,       \
                __LINE__);                                                     \
        exit(1);                                                               \
    }                                                                          \
}

#define CHECK_CURAND(call)                                                     \
{                                                                              \
    curandStatus_t err;                                                        \
    if ((err = (call)) != CURAND_STATUS_SUCCESS)                               \
    {                                                                          \
        fprintf(stderr, "Got CURAND error %d at %s:%d\n", err, __FILE__,       \
                __LINE__);                                                     \
        exit(1);                                                               \
    }                                                                          \
}

#define CHECK_CUFFT(call)                                                      \
{                                                                              \
    cufftResult err;                                                           \
    if ( (err = (call)) != CUFFT_SUCCESS)                                      \
    {                                                                          \
        fprintf(stderr, "Got CUFFT error %d at %s:%d\n", err, __FILE__,        \
                __LINE__);                                                     \
        exit(1);                                                               \
    }                                                                          \
}

#define CHECK_CUSPARSE(call)                                                   \
{                                                                              \
    cusparseStatus_t err;                                                      \
    if ((err = (call)) != CUSPARSE_STATUS_SUCCESS)                             \
    {                                                                          \
        fprintf(stderr, "Got error %d at %s:%d\n", err, __FILE__, __LINE__);   \
        cudaError_t cuda_err = cudaGetLastError();                             \
        if (cuda_err != cudaSuccess)                                           \
        {                                                                      \
            fprintf(stderr, "  CUDA error \"%s\" also detected\n",             \
                    cudaGetErrorString(cuda_err));                             \
        }                                                                      \
        exit(1);                                                               \
    }                                                                          \
}

inline double seconds() {
    struct timeval tp;
    struct timezone tzp;
    int i = gettimeofday(&tp, &tzp);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

inline void device_name() {
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));
}
#endif