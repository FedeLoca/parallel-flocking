#include "Boid.h"
#include "Flock.h"
#include "Utils.h"
#include <stdlib.h>
#include <time.h> 
#include <iostream> 
#include <map> 
#include <vector> 
#include <curand_kernel.h>

typedef struct BoidData {
    float p1;
    float p2;
    float p3;
    float d1;
    float d2;
    float d3;
} BoidData;

void computeNeighborhoods();

void updateFlock(float);
void getSeparationDirection(int, float*);
void getCohesionDirection(int, float*);
void getAlignDirection(int, float*);
void moveBoid(int, float);

void printNeighborhoods();
void printFlock();
void printBoid(int);