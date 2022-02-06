#include <vector>
#include <math.h>
#include <chrono>
#include <sys/time.h>

#ifndef UTILS_H
#define UTILS_H

typedef unsigned long ulong;
typedef unsigned int uint;

// returns the distance between the passed vectors
float vector3Distance(const float*, const float*);

// returns the sum of the passed vectors
void vector3Sum(const float*, const float*, float*);

// returns the subtraction of the passed vectors
void vector3Sub(const float*, const float*, float*);

// returns the multiplication of the passed vectors
void vector3Mul(const float*, const float, float*);

// returns the blending of the passed vectors representing directions
void blendDirections(const float*, const float*, const float*, float*);

// returns the magnitude of the passed vector
float vector3Magnitude(const float*);

// normalizes the passed vector
void vector3Normalize(float*);

inline double seconds() {
    struct timeval tp;
    struct timezone tzp;
    int i = gettimeofday(&tp, &tzp);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

#endif