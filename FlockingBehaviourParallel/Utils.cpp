#include "Utils.h"

float vector3Distance(const std::vector<float>& v, const std::vector<float>& w){
    return sqrt(pow(w[0] - v[0], 2) + pow(w[1] - v[1], 2) + pow(w[2] - v[2], 2));
}

std::vector<float> vector3Sum(const std::vector<float>& v, const std::vector<float>& w){
    return std::vector<float>{v[0] + w[0], v[1] + w[1], v[2] + w[2]};
}

std::vector<float> vector3Sub(const std::vector<float>& v, const std::vector<float>& w){
    return std::vector<float>{v[0] - w[0], v[1] - w[1], v[2] - w[2]};
}

std::vector<float> vector3Mul(const std::vector<float>& v, const float n){
    return std::vector<float>{v[0] * n, v[1] * n, v[2] * n};
}

std::vector<float> blendDirections(const std::vector<float>& v, const std::vector<float>& w, const std::vector<float>& u){
    
    std::vector<float> res{0,0,0};
    res = vector3Sum(res, v);
    res = vector3Sum(res, w);
    res = vector3Sum(res, u);

    return res;
}

float vector3Magnitude(const std::vector<float>& v){
    return sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
}

void vector3Normalize(std::vector<float>& v){
    if(v[0] != 0 || v[1] != 0 || v[2] != 0){
        float magnitude = vector3Magnitude(v);
        v[0] *= (1/magnitude);
        v[1] *= (1/magnitude);
        v[2] *= (1/magnitude);
    }
}