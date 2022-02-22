#include "FlockingBehaviour.h"

/*
 * Host function that computes distances between boids and fills the boolean matrix representing the neighborhoods
 * For future correctness check needs it uses as input the same data that will be used by the GPU (flockData and neighborhoods), but saves the results in the flockDataSeq array
 */
void computeNeighborhoods(bool* neighborhoodsSeq, float* flockData, int flockDim, float neighDim){

    float dist;
		for(int i = 0; i < flockDim; i++){
				for(int j = 0; j < flockDim; j++){
						
						//if the current cell is the upper right triangle of the matrix calculate the value
						if(j > i){	
								dist = vector3Distance(flockData+i, flockData+i+flockDim, flockData+i+2*flockDim, flockData+j, flockData+j+flockDim, flockData+j+2*flockDim);
								neighborhoodsSeq[i*flockDim+j] = (dist < neighDim);
						}
						else if(i == j){

								//if the current cell is on the diagonal the value is always zero because a boid is not neighbor of themself
								neighborhoodsSeq[i*flockDim+j] = 0;
						}
						else{
								
								//if the current cell is the lower left triangle of the matrix copy the corresponding value from the upper right triangle as the matrix is symmetric
								neighborhoodsSeq[i*flockDim+j] = neighborhoodsSeq[j*flockDim+i];
						}
				}
    }
}

/*
 * Device function for the computation of the separation component of one boid: the component is the normalized average repulsion vector from the neighbours
 */
void getSeparationDirection(int i, float* separation, bool* neighborhoodsSeq, float* flockData, int flockDim, float separationWeight){
    
		separation[0] = 0;
		separation[1] = 0;
		separation[2] = 0;

		//compute the average repulsion vector by summing the repulsion vectors
    float* tmp = (float*) malloc(3*sizeof(float));
		float magnitude;
    for(int j = 0; j < flockDim; j++){
				if(neighborhoodsSeq[i*flockDim+j]){
						
						//calculate the vector from the boid position to the neighbour position
						vector3Sub(flockData+i, flockData+i+flockDim, flockData+i+2*flockDim, flockData+j, flockData+j+flockDim, flockData+j+2*flockDim, tmp);

						//normalize it and divide it by its magnitude to obtain the repulsion vector
						magnitude = vector3Magnitude(tmp);
						vector3Normalize(tmp);
						vector3Mul(tmp, 1/(magnitude + 0.0001), tmp);

						//sum it to the current separation
						vector3Sum(separation, tmp, separation);
				}
    }

		//normalize and weight the component
    vector3Normalize(separation);
    vector3Mul(separation, separationWeight, separation);

    free(tmp);
}

/*
 * Host function for the computation of the cohesion component of one boid: the component is the normalized vector from the boid current position to the average position of its neighbours
 */
void getCohesionDirection(int i, float* cohesion, bool* neighborhoodsSeq, float* flockData, int flockDim, float cohesionWeight){
    
		cohesion[0] = 0;
		cohesion[1] = 0;
		cohesion[2] = 0;

		//compute the sum of all the neighbours positions
    float count = 0.0;
    for(int j = 0; j < flockDim; j++){
				if(neighborhoodsSeq[i*flockDim+j]){
						vector3Sum(cohesion, cohesion+1, cohesion+2, flockData+j, flockData+j+flockDim, flockData+j+2*flockDim, cohesion);
						count++;
				}
    }

		//calculate the average position and the vector to it only if there is at least one neighbours, otherwise the cohesion component remains the zero vector
    if(count != 0){
        vector3Mul(cohesion, 1.0/count, cohesion);
        vector3Sub(cohesion, cohesion+1, cohesion+2, flockData+i, flockData+i+flockDim, flockData+i+2*flockDim, cohesion);
    }

		//normalize and weight the component
    vector3Normalize(cohesion);
    vector3Mul(cohesion, cohesionWeight, cohesion);
}

/*
 * Host function for the computation of the align component of one boid: the component is the average direction of the neighbours
 */
void getAlignDirection(int i, float* align, bool* neighborhoodsSeq, float* flockData, int flockDim, float alignWeight){
		
		align[0] = 0;
		align[1] = 0;
		align[2] = 0;
    
		//compute the average direction by summing the neighbours directions
    for(int j = 0; j < flockDim; j++){
				if(neighborhoodsSeq[i*flockDim+j]){
        		vector3Sum(align, align+1, align+2, flockData+j+3*flockDim, flockData+j+4*flockDim, flockData+j+5*flockDim, align);
				}
    }

		//normalize and weight the component
    vector3Normalize(align);
    vector3Mul(align, alignWeight, align);
}

/*
 * Host function that updates the boid position by moving it towards its direction at the given velocity
 * For future correctness check needs it uses as input the same data that will be used by the GPU (flockData and neighborhoods), but saves the results in the flockDataSeq array
 */
void moveBoid(int i, float time, float* flockData, float* flockDataSeq, int flockDim) { 

    for(int j = 0; j < 3; j++){
        flockDataSeq[i+j*flockDim] = flockData[i+j*flockDim] + flockDataSeq[i+(j+3)*flockDim] * velocity * time;
    }
}

/*
 * Host function that determines the new direction of each boid based on its neighbors status and then determines each boid new position moving in the new direction at the given velocity.
 * For future correctness check needs it uses as input the same data that will be used by the GPU (flockData and neighborhoods), but saves the results in the flockDataSeq array
 */
void updateFlock(float time, bool* neighborhoodsSeq, float* flockData, float* flockDataSeq, int flockDim, int neighDim, float separationWeight, float cohesionWeight, float alignWeight){
    
    float* cohesion = (float*) malloc(3*sizeof(float));
    float* separation = (float*) malloc(3*sizeof(float));
    float* align = (float*) malloc(3*sizeof(float));
    float* finalDirection = (float*) malloc(3*sizeof(float));
		for(int i = 0; i < flockDim; i++){
      
				//calculate the components of the new direction of the boid
        getSeparationDirection(i, separation, neighborhoodsSeq, flockData, flockDim, separationWeight);
        getCohesionDirection(i, cohesion, neighborhoodsSeq, flockData, flockDim, cohesionWeight);
        getAlignDirection(i, align, neighborhoodsSeq, flockData, flockDim, alignWeight);

				//blend them togheter and update the direction of the boid if the resulting direction is not the zero vector
        blendDirections(separation, cohesion, align, finalDirection);
        if (finalDirection[0] != 0 || finalDirection[1] != 0 || finalDirection[2] != 0) {
            flockDataSeq[i+3*flockDim] = finalDirection[0];
						flockDataSeq[i+4*flockDim] = finalDirection[1];
						flockDataSeq[i+5*flockDim] = finalDirection[2];
        }
				else{
						
						//otherwise keep the current direction

						flockDataSeq[i+3*flockDim] = flockData[i+3*flockDim];
						flockDataSeq[i+4*flockDim] = flockData[i+4*flockDim];
						flockDataSeq[i+5*flockDim] = flockData[i+5*flockDim];
				}

				//update the position
        moveBoid(i, time, flockData, flockDataSeq, flockDim);
    }

    free(cohesion);
    free(separation);
    free(align);
    free(finalDirection);

		//computeNeighborhoods(neighborhoodsSeq, flockData, flockDim, neighDim);
}

/*
 * Host function that generates the positions and the directions of all boids in the flock
 */
void generateFlock(float* flockDataSeq, int numsToGenerate, int maxRand, int minRand, int div){
    
    for(int i = 0; i < numsToGenerate; i+=6){
			flockDataSeq[i] = (minRand + rand() % (maxRand + 1 - minRand)) / div;
			flockDataSeq[i+1] = (minRand + rand() % (maxRand + 1 - minRand)) / div;
			flockDataSeq[i+2] = (minRand + rand() % (maxRand + 1 - minRand)) / div;
			flockDataSeq[i+3] = (minRand + rand() % (maxRand + 1 - minRand)) / div;
			flockDataSeq[i+4] = (minRand + rand() % (maxRand + 1 - minRand)) / div;
			flockDataSeq[i+5] = (minRand + rand() % (maxRand + 1 - minRand)) / div;
		
			//normalize the direction
			vector3Normalize(flockDataSeq+i+3);
		}
}

/*
 * Host function that prints a single boid of the flock
 */
void printBoid(int i, float* flockData, int flockDim){
		
		std::cout << std::setprecision(4) << "Boid " << i << ": " << "pos(" << flockData[i] << ", " << flockData[i+flockDim] << ", " << flockData[i+flockDim*2] << 
		"); dir(" << flockData[i+flockDim*3] << ", " << flockData[i+flockDim*4] << ", " << flockData[i+flockDim*5] << ")" << std::endl;
    printf("Boid direction magnitude: %.5f\n", vector3Magnitude(flockData+i+flockDim*3, flockData+i+flockDim*4, flockData+i+flockDim*5));
}

/*
 * Host function that prints all the boids in the flock
 */
void printFlock(float* flockData, int flockDim){
    
    for(int i = 0; i < flockDim; i++){
				printBoid(i, flockData, flockDim);
    }
}

/*
 * Host function that prints all the neighbors of each boid
 */
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

/*
 * Host function for checking if the results of the neighborhoods computation made by the GPU and by the CPU is the same
 */
bool checkNeighborhoodsCorrectness(bool* neighborhoods, bool* neighborhoodsSeq, float* flockData, int flockDim){
    
    bool correct = 1;
		for(int i = 0; i < flockDim; i++){
				
				if(!correct){
						break;
				}

				for(int j = 0; j < flockDim; j++){

						//compare each value for which the distance from the boids was different from the neighbour dimension (in that cases the distance comparison may be have a different outcomes for
						//GPU and for CPU because of imprecision in the float comparison)
						if(neighborhoodsSeq[i*flockDim+j] != neighborhoods[i*flockDim+j] && 
						   vector3Distance(flockData+i, flockData+i+flockDim, flockData+i+2*flockDim, flockData+j, flockData+j+flockDim, flockData+j+2*flockDim) != neighDim){
								
								correct = 0;

								//print debug data if they are not equal
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

/*
 * Host function for checking if the results of the update made by the GPU and by the CPU is the same
 */
bool checkUpdateCorrectness(float* flockData, float* flockDataSeq, int flockDim){
		
		bool correct = 1;
		for(int i = 0; i < flockDim * 6; i++){

				//compare the float values with a very small tolerance
				if(fabs(flockData[i] - flockDataSeq[i]) > tolerance){
						
					  correct = 0;

						//print debug data if they are not equal
						printf("i: %i\n", i);
						printf("seq: %.5f --- par: %.5f\n", flockDataSeq[i], flockData[i]);

						break;
				}
    }

    return correct;
}