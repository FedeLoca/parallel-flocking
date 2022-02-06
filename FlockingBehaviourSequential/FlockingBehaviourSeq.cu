#include "FlockingBehaviourSeq.h"

float velocity = 20; // boid velocity in meters per second
double updateTime = 1.5; // update time of the simulation in seconds
float separationWeigth = 1; // weight of the separation component in the blending
float cohesionWeigth = 1; // weight of the cohesion component in the blending
float alignWeigth = 1; // weight of the align component in the blending
int flockDim = 10000000; // 10000000 number of boids in the flock
float neighDim = 75000000; // 75000000 dimension of the neighborhood in meters
int minRand = -50000; // -50000 minimum value that can be generated for initial position and direction
int maxRand = 50000; // 50000 maximum value that can be generated for initial position and direction
float decimals = 3; // 3 number of decimal digits in the generated values for initial position and direction
int iterations = 1; // number of updates 

float* flockData;
bool* neighborhoods;

int main(void) {

	srand (time(NULL));
	float div = pow(10.0, decimals);
	int numsToGenerate = flockDim * 6;

	// generate boids with random initial position and direction and add them to the flock
	printf("\n\nCPU Flock generation...\n");
	double cpuTimeStart = seconds();

	flockData = (float*) malloc(numsToGenerate * sizeof(float));
	for(int i = 0; i < numsToGenerate; i+=6){
			flockData[i] = (minRand + rand() % (maxRand + 1 - minRand)) / div;
			flockData[i+1] = (minRand + rand() % (maxRand + 1 - minRand)) / div;
			flockData[i+2] = (minRand + rand() % (maxRand + 1 - minRand)) / div;
			flockData[i+3] = (minRand + rand() % (maxRand + 1 - minRand)) / div;
			flockData[i+4] = (minRand + rand() % (maxRand + 1 - minRand)) / div;
			flockData[i+5] = (minRand + rand() % (maxRand + 1 - minRand)) / div;
		
			vector3Normalize(flockData+i+3);
	}

	double cpuTime = seconds() - cpuTimeStart;
	printf("    CPU elapsed time: %.5f (sec)\n", cpuTime);

	for(int i = 0; i < 12; i++){
			printf("first: %.5f\n", flockData[i]);
	}
	for(int i = 0; i < 12; i++){
			printf("last: %.5f\n", flockData[numsToGenerate - i - 1]);
	}

	// calculate neighborhoods (data stored in a boolean matrix)
  neighborhoods = (bool*) malloc(flockDim * flockDim * sizeof(bool));

	printf("\n\nCPU neighborhoods generation...\n");
	cpuTimeStart = seconds();
	computeNeighborhoods();
	cpuTime = seconds() - cpuTimeStart;
	printf("    CPU elapsed time: %.5f (sec)\n", cpuTime);

	//printFlock();		
	//std::cout << std::endl;
	//printNeighborhoods();

	// start simulation loop that updates the flock each updateTime
	double loopStart = seconds();
	double tmpTime = updateTime;
	while(iterations > 0){

	 		auto duration = seconds() - loopStart;
	 		if(duration >= tmpTime)
			{
				 	printf("\n\nCPU Flock update...\n");
					cpuTimeStart = seconds();
					updateFlock(updateTime);
					cpuTime = seconds() - cpuTimeStart;
					printf("    CPU elapsed time: %.5f (sec)", cpuTime);
				
					tmpTime += updateTime;
					iterations--;

					//std::cout << std::endl;
				  //printFlock();
				
					//std::cout << std::endl;
					//printNeighborhoods();
			}
	}

	free(flockData);
	free(neighborhoods);
	
	return 0;
}

void computeNeighborhoods(){
    
    for(int i = 0; i < flockDim; i++){
				for(int j = 0; j < flockDim; j++){
						if(j > i){	
								neighborhoods[i*flockDim+j] = vector3Distance(flockData+i*6, flockData+j*6) <= neighDim;
						}
						else if(i == j){
								neighborhoods[i*flockDim+j] = 0;
						}
						else{
								neighborhoods[i*flockDim+j] = neighborhoods[j*flockDim+i];
						}
				}
    }
}

void printFlock(){
    
    for(int i = 0; i < flockDim; i++){
				std::cout << std::setprecision(4) << "Boid " << i << ": " << "pos(" << flockData[i*6] << ", " << flockData[i*6+1] << ", " << flockData[i*6+2] << 
    		"); dir(" << flockData[i*6+3] << ", " << flockData[i*6+4] << ", " << flockData[i*6+5] << ")" << std::endl;
    }
}

void printNeighborhoods(){
    
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

void updateFlock(float time){
    
    float* cohesion = (float*) malloc(3*sizeof(float));
    float* separation = (float*) malloc(3*sizeof(float));
    float* align = (float*) malloc(3*sizeof(float));
    float* finalDirection = (float*) malloc(3*sizeof(float));
		for(int i = 0; i < flockDim; i++){
      
        getSeparationDirection(i, separation);
        getCohesionDirection(i, cohesion);
        getAlignDirection(i, align);

        blendDirections(separation, cohesion, align, finalDirection);
        if (finalDirection[0] != 0 || finalDirection[1] != 0 || finalDirection[2] != 0) {
            flockData[i*6+3] = finalDirection[0];
						flockData[i*6+4] = finalDirection[1];
						flockData[i*6+5] = finalDirection[2];
        }

        moveBoid(i, time);
    }

    free(cohesion);
    free(separation);
    free(align);
    free(finalDirection);

    computeNeighborhoods();
}

void getSeparationDirection(int i, float* separation){
    
    float* tmp = (float*) malloc(3*sizeof(float));
    for(int j = 0; j < flockDim; j++){
				if(neighborhoods[i*flockDim+j]){
						vector3Sub(flockData+i*6, flockData+j*6, tmp);
						vector3Normalize(tmp);
						vector3Mul(tmp, 1/(vector3Magnitude(tmp) + 0.0001), tmp);
						vector3Sum(separation, tmp, separation);
				}
    }

    vector3Normalize(separation);
    vector3Mul(separation, separationWeigth, separation);

    free(tmp);
}

void getCohesionDirection(int i, float* cohesion){
    
    float count = 0.0;
    for(int j = 0; j < flockDim; j++){
				if(neighborhoods[i*flockDim+j]){
						vector3Sum(cohesion, flockData+j*6, cohesion);
						count++;
				}
    }

    if(count != 0){
        vector3Mul(cohesion, 1.0/count, cohesion);
        vector3Sub(cohesion, flockData+i*6, cohesion);
    }

    vector3Normalize(cohesion);
    vector3Mul(cohesion, cohesionWeigth, cohesion);
}

void getAlignDirection(int i, float* align){
    
    for(int j = 0; j < flockDim; j++){
				if(neighborhoods[i*flockDim+j]){
        		vector3Sum(align, flockData+j*6+3, align);
				}
    }

    vector3Normalize(align);
    vector3Mul(align, alignWeigth, align);
}

void moveBoid(int i, float time) { 
    
    for(int j = 0; j < 3; j++){
        flockData[i*6+j] += flockData[i*6+3+j] * velocity * time;
    }
}