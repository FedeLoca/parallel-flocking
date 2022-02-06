#include "Flock.h"

Flock::Flock(): boidsMap{}, neighborhoodMap{} {}

Flock::Flock(std::map<int, Boid>&& boidsMap): boidsMap{boidsMap} {
    
    updateNeighborhoodMap();
}

const std::map<int, std::vector<int>>& Flock::getNeighborhoodMap() const { return neighborhoodMap; }

const std::map<int, Boid>& Flock::getBoidsMap() const { return boidsMap; }

const std::vector<int>& Flock::getNeighborhood(const Boid& b) const { return neighborhoodMap.at(b.getId()); }

void Flock::setNeighborhoodDim(int value){ neighborhoodDim = value; }

void Flock::setBlendingWeigths(const int s, const int c, const int a){
    separationWeigth = s;
    cohesionWeigth = c;
    alignWeigth = a;
}

std::vector<int> Flock::computeNeighborhood(const Boid& b){
    
    std::vector<int> neighborhood{};
    for(const auto& elem : boidsMap){
        
        if(elem.first != b.getId() && vector3Distance(elem.second.getPosition(), b.getPosition()) <= neighborhoodDim){
            neighborhood.push_back(elem.first);
        }
    }

    return neighborhood;
}

void Flock::addBoid(const Boid& b){
    boidsMap.insert(std::pair<int,Boid>(b.getId(), b));
    neighborhoodMap.insert(std::pair<int,std::vector<int>>(b.getId(), std::vector<int>{}));
}

void Flock::updateNeighborhoodMap(){
    
    for(const auto& elem : boidsMap){
        if(neighborhoodMap.find(elem.first) != neighborhoodMap.end()){
            neighborhoodMap[elem.first] = computeNeighborhood(elem.second);
        }
        else{
            neighborhoodMap.insert(std::pair<int,std::vector<int>>(elem.first, computeNeighborhood(elem.second)));
        }
    }
}

void Flock::updateFlock(float time){
    
    float* cohesion = (float*) malloc(3*sizeof(float));
    float* separation = (float*) malloc(3*sizeof(float));
    float* align = (float*) malloc(3*sizeof(float));
    float* finalDirection = (float*) malloc(3*sizeof(float));
		for(auto& elem : boidsMap){
      
        getSeparationDirection(elem.first, separation);
        getCohesionDirection(elem.first, cohesion);
        getAlignDirection(elem.first, align);

        blendDirections(separation, cohesion, align, finalDirection);
        if (finalDirection[0] != 0 || finalDirection[1] != 0 || finalDirection[2] != 0) {
            elem.second.setDirection(finalDirection);
        }

        elem.second.move(time);
    }

    free(cohesion);
    free(separation);
    free(align);
    free(finalDirection);

    updateNeighborhoodMap();
}

void Flock::getSeparationDirection(int b, float* separation) const{
    
    float* tmp = (float*) malloc(3*sizeof(float));
    for(const auto& n : neighborhoodMap.at(b)){
        vector3Sub(boidsMap.at(b).getPosition(), boidsMap.at(n).getPosition(), tmp);
        vector3Normalize(tmp);
        vector3Mul(tmp, 1/(vector3Magnitude(tmp) + 0.0001), tmp);
        vector3Sum(separation, tmp, separation);
    }

    vector3Normalize(separation);
    vector3Mul(separation, separationWeigth, separation);

    free(tmp);
}

void Flock::getCohesionDirection(int b, float* cohesion) const{
    
    float count = 0.0;
    for(const auto& n : neighborhoodMap.at(b)){
        vector3Sum(cohesion, boidsMap.at(n).getPosition(), cohesion);
        count++;
    }

    if(count != 0){
        vector3Mul(cohesion, 1.0/count, cohesion);
        vector3Sub(cohesion, boidsMap.at(b).getPosition(), cohesion);
    }

    vector3Normalize(cohesion);
    vector3Mul(cohesion, cohesionWeigth, cohesion);
}

void Flock::getAlignDirection(int b, float* align) const{
    
    for(const auto& n : neighborhoodMap.at(b)){
        vector3Sum(align, boidsMap.at(n).getDirection(), align);
    }

    vector3Normalize(align);
    vector3Mul(align, alignWeigth, align);
}

void Flock::print() const{
     
    for(const auto& elem : boidsMap){
        elem.second.print();
    }
}

void Flock::printNeighborhoods() const{
     
    for(const auto& elem : boidsMap){
        std::cout << elem.first << ": ";
        for(const auto& elem : getNeighborhood(elem.second)){
            std::cout << elem << ", ";
        }
        std::cout << std::endl;
    }
}