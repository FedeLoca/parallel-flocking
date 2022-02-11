#include "Utils.h"

float vector3Distance(const float* v, const float* w){
    return sqrt(pow(w[0] - v[0], 2) + pow(w[1] - v[1], 2) + pow(w[2] - v[2], 2));
}

float vector3Distance(const float* a, const float* b, const float* c, const float* x, const float* y, const float* z){
    return sqrt(pow(a[0] - x[0], 2) + pow(b[0] - y[0], 2) + pow(c[0] - z[0], 2));
}

void vector3Sum(const float* v, const float* w, float* res){
    
    res[0] = v[0] + w[0];
    res[1] = v[1] + w[1];
    res[2] = v[2] + w[2];
}

void vector3Sum(const float* a, const float* b, const float* c, const float* x, const float* y, const float* z, float* res){
    
    res[0] = a[0] + x[0];
    res[1] = b[0] + y[0];
    res[2] = c[0] + z[0];
}

void vector3Sub(const float* v, const float* w, float* res){
    
    res[0] = v[0] - w[0];
    res[1] = v[1] - w[1];
    res[2] = v[2] - w[2];
}

void vector3Sub(const float* a, const float* b, const float* c, const float* x, const float* y, const float* z, float* res){
    
    res[0] = a[0] - x[0];
    res[1] = b[0] - y[0];
    res[2] = c[0] - z[0];
}

void vector3Mul(const float* v, const float n, float* res){
    
    res[0] = v[0] * n;
    res[1] = v[1] * n;
    res[2] = v[2] * n;
}

void vector3Mul(const float* a, const float* b, const float* c, const float n, float* res){
    
    res[0] = a[0] * n;
    res[1] = b[0] * n;
    res[2] = c[0] * n;
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

float vector3Magnitude(const float* a, const float* b, const float* c){
    return sqrt(pow(a[0], 2) + pow(b[0], 2) + pow(c[0], 2));
}

void vector3Normalize(float* v){
    if(v[0] != 0 || v[1] != 0 || v[2] != 0){
        float magnitude = vector3Magnitude(v);
        v[0] *= (1/magnitude);
        v[1] *= (1/magnitude);
        v[2] *= (1/magnitude);
    }
}

void vector3Normalize(float* a, float* b, float* c){
    if(a[0] != 0 || b[0] != 0 || c[0] != 0){
        float magnitude = vector3Magnitude(a,b,c);
        a[0] *= (1/magnitude);
        b[0] *= (1/magnitude);
        c[0] *= (1/magnitude);
    }
}