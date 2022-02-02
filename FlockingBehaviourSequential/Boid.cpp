#include "Boid.h"

Boid::Boid(): id{0}, velocity{1}, direction{0,0,0}, position{0,0,0} {}

Boid::Boid(const int id, const float vel, const std::vector<float>& pos, const std::vector<float>& dir): id{id}, velocity{vel}, direction{dir}, position{pos} {}

const std::vector<float>& Boid::getDirection() const { return direction; }

const std::vector<float>& Boid::getPosition() const { return position; }

const int& Boid::getId() const { return id; }

void Boid::setDirection(const std::vector<float>& dir) { direction = dir; }

void Boid::setPosition(const std::vector<float>& pos) { position = pos; }

void Boid::move(float time) { 
    
    for(int i = 0; i < 3; i++){
        position[i] += direction[i] * velocity * time;
    }
}

void Boid::print() const{
    std::cout << std::setprecision(4) << "Boid " << id << ": " << "pos(" << position[0] << ", " << position[1] << ", " << position[2] << 
    "); dir(" << direction[0] << ", " << direction[1] << ", " << direction[2] << ")" << std::endl;
}