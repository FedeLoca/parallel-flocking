#include "FlockingBehaviourPar.h"

void computeNeighborhoods(bool* neighborhoodsSeq, float* flockData, int flockDim, float neighDim){

    float dist;
		for(int i = 0; i < flockDim; i++){
				for(int j = 0; j < flockDim; j++){
						if(j > i){	
								dist = vector3Distance(flockData+i, flockData+i+flockDim, flockData+i+2*flockDim, flockData+j, flockData+j+flockDim, flockData+j+2*flockDim);
								neighborhoodsSeq[i*flockDim+j] = (dist <= neighDim);
						}
						else if(i == j){
								neighborhoodsSeq[i*flockDim+j] = 0;
						}
						else{
								neighborhoodsSeq[i*flockDim+j] = neighborhoodsSeq[j*flockDim+i];
						}
				}
    }
}

void getSeparationDirection(int i, float* separation, bool* neighborhoodsSeq, float* flockData, int flockDim, float separationWeight){
    
    float* tmp = (float*) malloc(3*sizeof(float));
		float magnitude;
    for(int j = 0; j < flockDim; j++){
				if(neighborhoodsSeq[i*flockDim+j]){
						vector3Sub(flockData+i, flockData+i+flockDim, flockData+i+2*flockDim, flockData+j, flockData+j+flockDim, flockData+j+2*flockDim, tmp);
						magnitude = vector3Magnitude(tmp);
						vector3Normalize(tmp);
						vector3Mul(tmp, 1/(magnitude + 0.0001), tmp);
						vector3Sum(separation, tmp, separation);
				}
    }

    vector3Normalize(separation);
    vector3Mul(separation, separationWeight, separation);

    free(tmp);
}

void getCohesionDirection(int i, float* cohesion, bool* neighborhoodsSeq, float* flockData, int flockDim, float cohesionWeight){
    
    float count = 0.0;
    for(int j = 0; j < flockDim; j++){
				if(neighborhoodsSeq[i*flockDim+j]){
						vector3Sum(cohesion, cohesion+1, cohesion+2, flockData+j, flockData+j+flockDim, flockData+j+2*flockDim, cohesion);
						count++;
				}
    }

    if(count != 0){
        vector3Mul(cohesion, 1.0/count, cohesion);
        vector3Sub(cohesion, cohesion+1, cohesion+2, flockData+i, flockData+i+flockDim, flockData+i+2*flockDim, cohesion);
    }

    vector3Normalize(cohesion);
    vector3Mul(cohesion, cohesionWeight, cohesion);
}

void getAlignDirection(int i, float* align, bool* neighborhoodsSeq, float* flockData, int flockDim, float alignWeight){
    
    for(int j = 0; j < flockDim; j++){
				if(neighborhoodsSeq[i*flockDim+j]){
        		vector3Sum(align, align+1, align+2, flockData+j+3*flockDim, flockData+j+4*flockDim, flockData+j+5*flockDim, align);
				}
    }

    vector3Normalize(align);
    vector3Mul(align, alignWeight, align);
}

void moveBoid(int i, float time, float* flockData, int flockDim) { 
    
    for(int j = 0; j < 3; j++){
        flockData[i+j*flockDim] += flockData[i+(j+3)*flockDim] * velocity * time;
    }
}

void updateFlock(float time, bool* neighborhoodsSeq, float* flockData, int flockDim, int neighDim, float separationWeight, float cohesionWeight, float alignWeight){
    
    float* cohesion = (float*) malloc(3*sizeof(float));
    float* separation = (float*) malloc(3*sizeof(float));
    float* align = (float*) malloc(3*sizeof(float));
    float* finalDirection = (float*) malloc(3*sizeof(float));
		for(int i = 0; i < flockDim; i++){
      
        getSeparationDirection(i, separation, neighborhoodsSeq, flockData, flockDim, separationWeight);
        getCohesionDirection(i, cohesion, neighborhoodsSeq, flockData, flockDim, cohesionWeight);
        getAlignDirection(i, align, neighborhoodsSeq, flockData, flockDim, alignWeight);

        blendDirections(separation, cohesion, align, finalDirection);
        if (finalDirection[0] != 0 || finalDirection[1] != 0 || finalDirection[2] != 0) {
            flockData[i+3*flockDim] = finalDirection[0];
						flockData[i+4*flockDim] = finalDirection[1];
						flockData[i+5*flockDim] = finalDirection[2];
        }

        moveBoid(i, time, flockData, flockDim);
    }

    free(cohesion);
    free(separation);
    free(align);
    free(finalDirection);

    computeNeighborhoods(neighborhoodsSeq, flockData, flockDim, neighDim);
}

void generateFlock(float* flockDataSeq, int numsToGenerate, int maxRand, int minRand, int div){
    
    for(int i = 0; i < numsToGenerate; i+=6){
			flockDataSeq[i] = (minRand + rand() % (maxRand + 1 - minRand)) / div;
			flockDataSeq[i+1] = (minRand + rand() % (maxRand + 1 - minRand)) / div;
			flockDataSeq[i+2] = (minRand + rand() % (maxRand + 1 - minRand)) / div;
			flockDataSeq[i+3] = (minRand + rand() % (maxRand + 1 - minRand)) / div;
			flockDataSeq[i+4] = (minRand + rand() % (maxRand + 1 - minRand)) / div;
			flockDataSeq[i+5] = (minRand + rand() % (maxRand + 1 - minRand)) / div;
		
			vector3Normalize(flockDataSeq+i+3);
		}
}

void printBoid(int i, float* flockData, int flockDim){
		
		std::cout << std::setprecision(4) << "Boid " << i << ": " << "pos(" << flockData[i] << ", " << flockData[i+flockDim] << ", " << flockData[i+flockDim*2] << 
		"); dir(" << flockData[i+flockDim*3] << ", " << flockData[i+flockDim*4] << ", " << flockData[i+flockDim*5] << ")" << std::endl;
    printf("Boid direction magnitude: %.5f\n", vector3Magnitude(flockData+i+flockDim*3, flockData+i+flockDim*4, flockData+i+flockDim*5));
}

void printFlock(float* flockData, int flockDim){
    
    for(int i = 0; i < flockDim; i++){
				printBoid(i, flockData, flockDim);
    }
}

void printNeighborhoods(bool* neighborhoodsSeq, int flockDim){
    
    for(int i = 0; i < flockDim; i++){
				std::cout << i << ": ";
				for(int j = 0; j < flockDim; j++){
						if(neighborhoods[i*flockDim+j]){
								std::cout << j << ", ";
						}
				}
				std::cout << std::endl;
    }
}

bool checkNeighborhoodsCorrectness(bool* neighborhoods, bool* neighborhoodsSeq, float* flockData, int flockDim){
    
    bool correct = 1;
		for(int i = 0; i < flockDim; i++){
				
				if(!correct){
						break;
				}

				for(int j = 0; j < flockDim; j++){
						if(neighborhoodsSeq[i*flockDim+j] != neighborhoods[i*flockDim+j] && vector3Distance(flockData+i, flockData+i+flockDim, flockData+i+2*flockDim, flockData+j, flockData+j+flockDim, flockData+j+2*flockDim) != neighDim){
								correct = 0;

                printf("i: %i --- j: %i\n", i,j);
								printf("seq: %i, %.5f --- par: %i, %.5f\n", neighborhoodsSeq[i*flockDim+j], vector3Distance(flockData+i, flockData+i+flockDim, flockData+i+2*flockDim, flockData+j, flockData+j+flockDim, flockData+j+2*flockDim),
								        neighborhoods[i*flockDim+j], sqrt(pow(flockData[i] - flockData[j], 2) + pow(flockData[i+flockDim] - flockData[j+flockDim], 2) + pow(flockData[i+2*flockDim] - flockData[j+2*flockDim], 2)));
								
								printBoid(i, flockData, flockDim);
								printBoid(j, flockData, flockDim);

								break;
						}
				}
    }

    return correct;
}