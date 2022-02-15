#include "Utils.h"
#include <stdlib.h>
#include <time.h> 
#include <iostream> 
#include <iomanip>
#include <curand_kernel.h>

#ifndef FLOCKING_H
#define FLOCKING_H

extern float velocity;
extern double updateTime;
extern float separationWeight;
extern float cohesionWeight;
extern float alignWeight;
extern int flockDim;
extern float neighDim;
extern float tolerance;
extern int minRand;
extern int maxRand;
extern float decimals;
extern int iterations;
extern int generationsPerThread;

extern float* flockData;
extern bool* neighborhoods;
extern bool* neighborhoodsSeq;

typedef struct BoidData {
    float p1;
    float p2;
    float p3;
    float d1;
    float d2;
    float d3;
} BoidData;

void generateFlock(float*, int, int, int, int);

void computeNeighborhoods(bool*, float*, int, float);
bool checkNeighborhoodsCorrectness(bool*, bool*, float*, int);

void updateFlock(float, bool*, float*, int, int, float, float, float);
void getSeparationDirection(int, float*, bool*, float*, int, float*);
void getCohesionDirection(int, float*, bool*, float*, int, float*);
void getAlignDirection(int, float*, bool*, float*, int, float*);
void moveBoid(int, float, float*, int);

void printNeighborhoods(bool*, int);
void printFlock(float*, int);
void printBoid(int, float*, int);
#endif