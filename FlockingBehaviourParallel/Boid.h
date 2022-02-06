#include <vector>
#include <iostream>
#include <iomanip>
#include <string>

#ifndef BOID_H
#define BOID_H
class Boid{
    
    public:
    Boid();

    Boid(const int, const float, float*, float*);

    float* getDirection() const;
    float* getPosition() const;
    const int& getId() const;
    void setDirection(float*);
    void setPosition(float*);

    // move the boid by updating its position based on its current direction, velocity and time passed since the last update
    void move(float);

    void print() const;

    private:
    int id; // unique id of the boid
    float velocity; // boid velocity in meters per second
    float* position; // one unit is one meter
    float* direction; // normalized vector
};
#endif