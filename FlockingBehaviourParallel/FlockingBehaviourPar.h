#include "Boid.h"
#include "Flock.h"
#include "Utils.h"
#include <stdlib.h>
#include <time.h> 
#include <iostream> 
#include <map> 
#include <vector> 
#include <curand_kernel.h>

typedef struct BoidsData
{
    float** p1;
    float** p2;
    float** p3;
    float** d1;
    float** d2;
    float** d3;
} BoidsData;