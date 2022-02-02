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
    
    std::vector<float> cohesion;
    std::vector<float> separation;
    std::vector<float> align;
    std::vector<float> finalDirection;
		for(auto& elem : boidsMap){
      
        separation = getSeparationDirection(elem.first);
        cohesion = getCohesionDirection(elem.first);
        align = getAlignDirection(elem.first);

        finalDirection = blendDirections(separation, cohesion, align);
        if (finalDirection[0] != 0 || finalDirection[1] != 0 || finalDirection[2] != 0) {
            elem.second.setDirection(finalDirection);
        }

        elem.second.move(time);
    }

    updateNeighborhoodMap();
}

std::vector<float> Flock::getSeparationDirection(int b) const{
    
    std::vector<float> separation{0,0,0};
    std::vector<float> tmp;
    for(const auto& n : neighborhoodMap.at(b)){
        tmp = vector3Sub(boidsMap.at(b).getPosition(), boidsMap.at(n).getPosition());
        vector3Normalize(tmp);
        tmp = vector3Mul(tmp, 1/(vector3Magnitude(tmp) + 0.0001));
        separation = vector3Sum(separation, tmp);
    }

    vector3Normalize(separation);
    return vector3Mul(separation, separationWeigth);
}

std::vector<float> Flock::getCohesionDirection(int b) const{
    
    std::vector<float> cohesion{0,0,0};
    float count = 0.0;
    for(const auto& n : neighborhoodMap.at(b)){
        cohesion = vector3Sum(cohesion, boidsMap.at(n).getPosition());
        count++;
    }

    if(count != 0){
        cohesion = vector3Mul(cohesion, 1.0/count);
        cohesion = vector3Sub(cohesion, boidsMap.at(b).getPosition());
    }

    vector3Normalize(cohesion);
    return vector3Mul(cohesion, cohesionWeigth);
}

std::vector<float> Flock::getAlignDirection(int b) const{
    
    std::vector<float> align{0,0,0};
    for(const auto& n : neighborhoodMap.at(b)){
        align = vector3Sum(align, boidsMap.at(n).getDirection());
    }

    vector3Normalize(align);
    return vector3Mul(align, alignWeigth);
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