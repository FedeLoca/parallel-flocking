#include <vector>
#include <math.h>
#include <chrono>
#include <sys/time.h>

#ifndef UTILS_H
#define UTILS_H

typedef unsigned long ulong;
typedef unsigned int uint;

// returns the distance between the passed vectors
float vector3Distance(const std::vector<float>&, const std::vector<float>&);

// returns the sum of the passed vectors
std::vector<float> vector3Sum(const std::vector<float>&, const std::vector<float>&);

// returns the subtraction of the passed vectors
std::vector<float> vector3Sub(const std::vector<float>&, const std::vector<float>&);

// returns the multiplication of the passed vectors
std::vector<float> vector3Mul(const std::vector<float>&, const float);

// returns the blending of the passed vectors representing directions
std::vector<float> blendDirections(const std::vector<float>&, const std::vector<float>&, const std::vector<float>&);

// returns the magnitude of the passed vector
float vector3Magnitude(const std::vector<float>&);

// normalizes the passed vector
void vector3Normalize(std::vector<float>&);

inline double seconds() {
    struct timeval tp;
    struct timezone tzp;
    int i = gettimeofday(&tp, &tzp);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

#endif