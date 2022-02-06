#include "FlockingBehaviourPar.h"

#define GEN_BLOCK_SIZE 128
#define GEN_GRID_SIZE 32
//384  128
//64   32
#define NEIGH_GRID_SIZE 128

float velocity = 20; // boid velocity in meters per second
double updateTime = 1.5; // update time of the simulation in seconds
float separationWeigth = 1; // weight of the separation component in the blending
float cohesionWeigth = 1; // weight of the cohesion component in the blending
float alignWeigth = 1; // weight of the align component in the blending
int flockDim = 10; // 10000000 number of boids in the flock
float neighDim = 10; // 75000000 dimension of the neighborhood in meters
int minRand = -50000; // -50000 minimum value that can be generated for initial position and direction
int maxRand = 50000; // 50000 maximum value that can be generated for initial position and direction
float decimals = 3; // 0 number of decimal digits in the generated values for initial position and direction
int iterations = 1; // number of updates
int generationsPerThread = 1; //2500 number of boids a thread must generate

__constant__ float velocityDev;
__constant__ double updateTimeDev;
__constant__ float separationWeigthDev;
__constant__ float cohesionWeigthDev;
__constant__ float alignWeigthDev;
__constant__ int flockDimDev;
__constant__ float neighDimDev;
__constant__ int minRandDev;
__constant__ int minMaxDiffDev;
__constant__ float divDev;
__constant__ int threadsNumDev;
__constant__ int generationsPerThreadDev;
__constant__ int lastThreadGenerationsDev;

float* flockData;
bool* neighborhoods;

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

void printBoid(int i){
		
		std::cout << std::setprecision(4) << "Boid " << i << ": " << "pos(" << flockData[i] << ", " << flockData[i+flockDim] << ", " << flockData[i+flockDim*2] << 
		"); dir(" << flockData[i+flockDim*3] << ", " << flockData[i+flockDim*4] << ", " << flockData[i+flockDim*5] << ")" << std::endl;
}

void printFlock(){
    
    for(int i = 0; i < flockDim; i++){
				std::cout << std::setprecision(4) << "Boid " << i << ": " << "pos(" << flockData[i] << ", " << flockData[i+flockDim] << ", " << flockData[i+flockDim*2] << 
				"); dir(" << flockData[i+flockDim*3] << ", " << flockData[i+flockDim*4] << ", " << flockData[i+flockDim*5] << ")" << std::endl;
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

/*
 * Initialize cuRAND states
 */
__global__ void initializeStates(uint seed, curandState* states) {

		uint tid = threadIdx.x + blockDim.x * blockIdx.x;

		curand_init(seed, tid, 0, &states[tid]);
}

/*
 * Kernel for flock generation: generates boids with random initial position and direction
 */
__global__ void generateBoidsStatus(float* generated, curandState* states) {

	uint tid = threadIdx.x + blockDim.x * blockIdx.x;

	if(tid < threadsNumDev){
			
			curandState localState = states[tid];

			//to avoid accesses out of bounds
			//if the thread is the last make it generate the numbers for the remaining boids, otherwise make it generate generationsPerThread numbers
			int myGenerations = (tid == (threadsNumDev - 1)) * lastThreadGenerationsDev + !(tid == (threadsNumDev - 1)) * generationsPerThreadDev;
	
			float d1, d2, d3;
			float value;
			uint pos;
			//myGenerations = myGenerations/2;
			for(uint i = 0; i < myGenerations; i++){
					
					d1 = (curand_uniform(&localState) * minMaxDiffDev + minRandDev) / divDev;
					d2 = (curand_uniform(&localState) * minMaxDiffDev + minRandDev) / divDev;
					d3 = (curand_uniform(&localState) * minMaxDiffDev + minRandDev) / divDev;

					value = sqrt(d1*d1 + d2*d2 + d3*d3);

					//if the magnitude is 0 avoid dividing for 0 and set the value to 0, otherwise calculate the value to normalize
					value = !(value == 0) * 1/(value + (value == 0));

					pos = tid + i * threadsNumDev;
					generated[pos] = (curand_uniform(&localState) * minMaxDiffDev + minRandDev) / divDev;
					generated[pos + flockDimDev] = (curand_uniform(&localState) * minMaxDiffDev + minRandDev) / divDev;
					generated[pos + 2 * flockDimDev] = (curand_uniform(&localState) * minMaxDiffDev + minRandDev) / divDev;
					generated[pos + 3 * flockDimDev] = d1 * value;
					generated[pos + 4 * flockDimDev] = d2 * value;
					generated[pos + 5 * flockDimDev] = d3 * value;
			}		
			
			states[tid] = localState;
	}
}

/*
 * Kernel for neighborhoods computation: computes distances and fills the boolean matrix
 */
__global__ void computeAllNeighborhoods(float* flockData, bool* neighborhoods) {

		uint tid = threadIdx.x + blockDim.x * blockIdx.x;

		if(tid < threadsNumDev){
				
				bool value;
				for(int i = 0; i < flockDimDev; i++){

						if(tid == i){
								neighborhoods[i*flockDimDev+tid] = 0;
						}
						else{
								value = sqrt(pow(flockData[tid*6] - flockData[i*6], 2) + pow(flockData[tid*6+1] - flockData[i*6+1], 2) + pow(flockData[tid*6+2] - flockData[i*6+2], 2)) <= neighDimDev;
								neighborhoods[i*flockDimDev+tid] = value;
						}
						
						/*
						value = sqrt(pow(flockData[tid*6] - flockData[i*6], 2) + pow(flockData[tid*6+1] - flockData[i*6+1], 2) + pow(flockData[tid*6+2] - flockData[i*6+2], 2)) <= neighDimDev;
						//value = (tid == i) * 0 + !(tid == i) * value;
				    neighborhoods[i*flockDimDev+tid] = value;
						//neighborhoods[tid*flockDimDev+i] = value;
						*/
				}
		}
}

int main(void) {

		device_name();

		// events to measure time
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		float milliseconds = 0;

		// prepare for flock generation
		int minMaxDiff = maxRand - minRand;
		float div = pow(10.0, decimals);
		int numsToGenerate = flockDim * 6;

		float* flockData;
		curandState* states;
		int blockSize = GEN_BLOCK_SIZE;
		int gridSize = GEN_GRID_SIZE; 
		CHECK(cudaMallocManaged((void **) &flockData, numsToGenerate * sizeof(float)));
		CHECK(cudaMalloc((void **) &states, blockSize * gridSize * sizeof(curandState)));
		int threadsNum = (flockDim + generationsPerThread - 1) / generationsPerThread;
		int lastThreadGenerations = flockDim - generationsPerThread * (threadsNum - 1);

		// initialize all constant values
		cudaMemcpyToSymbol(velocityDev, &velocity, sizeof(velocityDev));
		cudaMemcpyToSymbol(updateTimeDev, &updateTime, sizeof(updateTimeDev));
		cudaMemcpyToSymbol(separationWeigthDev, &separationWeigth, sizeof(separationWeigthDev));
		cudaMemcpyToSymbol(cohesionWeigthDev, &cohesionWeigth, sizeof(cohesionWeigthDev));
		cudaMemcpyToSymbol(alignWeigthDev, &alignWeigth, sizeof(alignWeigthDev));
		cudaMemcpyToSymbol(flockDimDev, &flockDim, sizeof(flockDimDev));
		cudaMemcpyToSymbol(neighDimDev, &neighDim, sizeof(neighDimDev));
		cudaMemcpyToSymbol(minRandDev, &minRand, sizeof(minRandDev));
		cudaMemcpyToSymbol(minMaxDiffDev, &minMaxDiff, sizeof(minMaxDiffDev));
		cudaMemcpyToSymbol(divDev, &div, sizeof(divDev));
		cudaMemcpyToSymbol(threadsNumDev, &threadsNum, sizeof(threadsNumDev));
		cudaMemcpyToSymbol(generationsPerThreadDev, &generationsPerThread, sizeof(generationsPerThreadDev));
		cudaMemcpyToSymbol(lastThreadGenerationsDev, &lastThreadGenerations, sizeof(lastThreadGenerationsDev));

		printf("\nthreadsNum: %i\n", threadsNum);
		printf("effthreadsNum: %i\n", blockSize * gridSize);
		printf("genperthread: %i\n", generationsPerThread);
		if(threadsNum > blockSize * gridSize){
				std::cout << "\nNot enough threads" << std::endl;
		}

		printf("\n\nGPU Flock generation...\n");
		cudaEventRecord(start);

		// generate flock

		initializeStates<<<gridSize, blockSize>>>(time(NULL), states);

		generateBoidsStatus<<<gridSize, blockSize>>>(flockData, states);

		cudaEventRecord(stop);
		CHECK(cudaEventSynchronize(stop));
		cudaEventElapsedTime(&milliseconds, start, stop);
		printf("    GPU elapsed time: %.5f (sec)\n", milliseconds / 1000);

		/* 
		// print first and last data to see if they are correct
		for(int i = 0; i < 12; i++){
				printf("first gen: %.5f\n", flockData[i]);
		}
		for(int i = 0; i < 12; i++){
				printf("last gen: %.5f\n", flockData[numsToGenerate - i - 1]);
		}
		*/

		std::cout << std::setprecision(4) << "Boid " << 0 << ": " << "pos(" << flockData[0] << ", " << flockData[0+flockDim] << ", " << flockData[0+flockDim*2] << 
			"); dir(" << flockData[0+flockDim*3] << ", " << flockData[0+flockDim*4] << ", " << flockData[0+flockDim*5] << ")" << std::endl;

		std::cout << std::setprecision(4) << "Boid " << flockDim-1 << ": " << "pos(" << flockData[flockDim-1] << ", " << flockData[flockDim-1+flockDim] << ", " << flockData[flockDim-1+flockDim*2] << 
			"); dir(" << flockData[flockDim-1+flockDim*3] << ", " << flockData[flockDim-1+flockDim*4] << ", " << flockData[flockDim-1+flockDim*5] << ")" << std::endl;

		//printBoid(0);
		//printBoid(flockDim-1);

		CHECK(cudaFree(states));

		// neighborhoods data stored in a boolean matrix
		CHECK(cudaMallocManaged((void **) &neighborhoods, flockDim * flockDim * sizeof(bool)));
		threadsNum = flockDim; // flockDim/2;
		gridSize = NEIGH_GRID_SIZE;
		blockSize = (threadsNum + gridSize - 1)/gridSize;

		printf("\nthreadsNum: %i\n", threadsNum);
		printf("blocSize: %i\n", blockSize);
		printf("effthreadsNum: %i\n", blockSize * gridSize);
		if(threadsNum > blockSize * gridSize){
				std::cout << "\nNot enough threads" << std::endl;
		}

		cudaMemcpyToSymbol(threadsNumDev, &threadsNum, sizeof(threadsNumDev));

		printf("\n\nGPU Neighborhoods computation...\n");
		cudaEventRecord(start);

		computeAllNeighborhoods<<<gridSize, blockSize>>>(flockData, neighborhoods);

		cudaEventRecord(stop);
		CHECK(cudaEventSynchronize(stop));
		cudaEventElapsedTime(&milliseconds, start, stop);
		printf("    GPU elapsed time: %.5f (sec)\n", milliseconds / 1000);

		for(int i = 0; i < 12; i++){
				printf("first neigh: %d\n", neighborhoods[i]);
		}
		for(int i = 0; i < 12; i++){
				printf("last neigh: %d\n", neighborhoods[(flockDim-1)*flockDim+flockDim-1-i]);
		}

		for(int i = 0; i < flockDim; i++){
				std::cout << std::setprecision(4) << "Boid " << i << ": " << "pos(" << flockData[i] << ", " << flockData[i+flockDim] << ", " << flockData[i+flockDim*2] << 
				"); dir(" << flockData[i+flockDim*3] << ", " << flockData[i+flockDim*4] << ", " << flockData[i+flockDim*5] << ")" << std::endl;
    }

		//printFlock();		
		//std::cout << std::endl;
		printNeighborhoods();
		std::cout << std::endl;
		std::cout << std::endl;

		bool* neighborhoodsSeq = (bool*) malloc(flockDim * flockDim * sizeof(bool));

		for(int i = 0; i < flockDim; i++){
				for(int j = 0; j < flockDim; j++){
						if(j > i){	
								neighborhoodsSeq[i*flockDim+j] = vector3Distance(flockData+i*6, flockData+j*6) <= neighDim;
						}
						else if(i == j){
								neighborhoodsSeq[i*flockDim+j] = 0;
						}
						else{
								neighborhoodsSeq[i*flockDim+j] = neighborhoodsSeq[j*flockDim+i];
						}
				}
    }
		
		for(int i = 0; i < flockDim; i++){
				std::cout << i << ": ";
				for(int j = 0; j < flockDim; j++){
						if(neighborhoodsSeq[i*flockDim+j]){
								std::cout << j << ", ";
						}
				}
				std::cout << std::endl;
    }

		// start simulation loop that updates the flock each updateTime
		double loopStart = seconds();
		double tmpTime = updateTime;
		while(iterations > 0){

				auto duration = seconds() - loopStart;
				if(duration >= tmpTime)
				{
						printf("\n\nGPU Flock update...\n");
						double gpuTimeStart = seconds();
						updateFlock(updateTime);
						double gpuTime = seconds() - gpuTimeStart;
						printf("    GPU elapsed time: %.5f (sec)\n", gpuTime);
					
						tmpTime += updateTime;
						iterations--;

						//std::cout << std::endl;
						//printFlock();
					
						//std::cout << std::endl;
						//printNeighborhoods();
				}
		}

		CHECK(cudaFree(flockData));
		free(neighborhoods);
		
		return 0;
}