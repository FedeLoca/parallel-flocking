#include <iostream>
#include <map>
#include <vector>
#include "Boid.h"
#include "Utils.h"

#ifndef FLOCK_H
#define FLOCK_H
class Flock{
    
    public:
    Flock();
    Flock(std::map<int, Boid>&&);

    const std::map<int, std::vector<int>>& getNeighborhoodMap() const;
    const std::map<int, Boid>& getBoidsMap() const;

    const std::vector<int>& getNeighborhood(const Boid&) const;
    void setNeighborhoodDim(const int);
    void setBlendingWeigths(const int, const int, const int);

    // returns the list of boids that are nearer than neighborhoodDim from the passed boid
    std::vector<int> computeNeighborhood(const Boid&);

    // adds a boid to the flock
    void addBoid(const Boid&);

    // computes the neighborhoods of each boid based on the current situation and puts them inside the neighborhoodMap
    void updateNeighborhoodMap();

    // for each boid computes all the components that influence its direction, blends them and set the resulting 
    // direction as the current direction of the boid only if it is not (0,0,0). otherwise the old direction is maintained
    void updateFlock(float);

    void getSeparationDirection(const int, float*) const;
    void getCohesionDirection(const int, float*) const;
    void getAlignDirection(const int, float*) const;

    void print() const;
    void printNeighborhoods() const;

    private:
    std::map<int, std::vector<int>> neighborhoodMap; // associates the neighborhood of a boid to the id of the boid
    std::map<int, Boid> boidsMap; // associates a boid to its id
    float neighborhoodDim; // dimension of the neighborhood in meters
    float separationWeigth; // weight of the separation component in the blending
    float cohesionWeigth; // weight of the cohesion component in the blending
    float alignWeigth; // weight of the align component in the blending
};
#endif