#include "Utils.h"

float vector3Distance(const float* v, const float* w){
    return sqrt(pow(w[0] - v[0], 2) + pow(w[1] - v[1], 2) + pow(w[2] - v[2], 2));
}

void vector3Sum(const float* v, const float* w, float* res){
    
    res[0] = v[0] + w[0];
    res[1] = v[1] + w[1];
    res[2] = v[2] + w[2];
}

void vector3Sub(const float* v, const float* w, float* res){
    
    res[0] = v[0] - w[0];
    res[1] = v[1] - w[1];
    res[2] = v[2] - w[2];
}

void vector3Mul(const float* v, const float n, float* res){
    
    res[0] = v[0] * n;
    res[1] = v[1] * n;
    res[2] = v[2] * n;
}

void blendDirections(const float* v, const float* w, const float* u, float* res){
    
    res[0] = 0;
    res[1] = 0;
    res[2] = 0;
    vector3Sum(res, v, res);
    vector3Sum(res, w, res);
    vector3Sum(res, u, res);
}

float vector3Magnitude(const float* v){
    return sqrt(pow(v[0], 2) + pow(v[1], 2) + pow(v[2], 2));
}

void vector3Normalize(float* v){
    if(v[0] != 0 || v[1] != 0 || v[2] != 0){
        float magnitude = vector3Magnitude(v);
        v[0] *= (1/magnitude);
        v[1] *= (1/magnitude);
        v[2] *= (1/magnitude);
    }
}