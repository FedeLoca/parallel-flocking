#include "Boid.h"

Boid::Boid(const int id, const float vel, float* pos, float* dir): id{id}, velocity{vel}, direction{dir}, position{pos} {}

float* Boid::getDirection() const { return direction; }

float* Boid::getPosition() const { return position; }

const int& Boid::getId() const { return id; }

void Boid::setDirection(float* dir) { 
    direction[0] = dir[0]; 
    direction[1] = dir[1]; 
    direction[2] = dir[2]; 
}

void Boid::setPosition(float* pos) { 
    position[0] = pos[0];
    position[1] = pos[1];
    position[2] = pos[2];
}

void Boid::move(float time) { 
    
    for(int i = 0; i < 3; i++){
        position[i] += direction[i] * velocity * time;
    }
}

void Boid::print() const{
    std::cout << std::setprecision(4) << "Boid " << id << ": " << "pos(" << position[0] << ", " << position[1] << ", " << position[2] << 
    "); dir(" << direction[0] << ", " << direction[1] << ", " << direction[2] << ")" << std::endl;
}