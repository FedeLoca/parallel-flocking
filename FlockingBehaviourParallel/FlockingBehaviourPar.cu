#include "FlockingBehaviourPar.h"

#define GEN_BLOCK_SIZE 128
#define GEN_GRID_SIZE 32
//384  128
//64   32
#define NEIGH_BLOCK_SIZE 256
//128
#define UPDATE_BLOCK_SIZE 256

float velocity = 20; // boid velocity in meters per second
double updateTime = 1.5; // update time of the simulation in seconds
float separationWeigth = 1; // weight of the separation component in the blending
float cohesionWeigth = 1; // weight of the cohesion component in the blending
float alignWeigth = 1; // weight of the align component in the blending
int flockDim = 10240; // 10000000 number of boids in the flock
float neighDim = 50; // 50 dimension of the neighborhood in meters
float tolerance = 0.1f; //tolerance for float comparison
int minRand = -50000; // -50000 minimum value that can be generated for initial position and direction
int maxRand = 50000; // 50000 maximum value that can be generated for initial position and direction
float decimals = 3; // 3 number of decimal digits in the generated values for initial position and direction
int iterations = 1; // number of updates
int generationsPerThread = 250; //2500 number of boids a thread must generate

__constant__ float velocityDev;
__constant__ double updateTimeDev;
__constant__ float separationWeigthDev;
__constant__ float cohesionWeigthDev;
__constant__ float alignWeigthDev;
__constant__ int flockDimDev;
__constant__ float neighDimDev;
__constant__ float toleranceDev;
__constant__ int minRandDev;
__constant__ int minMaxDiffDev;
__constant__ float divDev;
__constant__ int threadsNumDev;
__constant__ int generationsPerThreadDev;
__constant__ int lastThreadGenerationsDev;

float* flockData;
bool* neighborhoods;
bool* neighborhoodsSeq;

void computeNeighborhoods(){
    
    for(int i = 0; i < flockDim; i++){
				for(int j = 0; j < flockDim; j++){
						if(j > i){	
								neighborhoodsSeq[i*flockDim+j] = vector3Distance(flockData+i, flockData+i+flockDim, flockData+i+2*flockDim, flockData+j, flockData+j+flockDim, flockData+j+2*flockDim) <= neighDim;
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
            flockData[i+3*flockDim] = finalDirection[0];
						flockData[i+4*flockDim] = finalDirection[1];
						flockData[i+5*flockDim] = finalDirection[2];
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
				if(neighborhoodsSeq[i*flockDim+j]){
						vector3Sub(flockData+i, flockData+i+flockDim, flockData+i+2*flockDim, flockData+j, flockData+j+flockDim, flockData+j+2*flockDim, tmp);
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
						vector3Sum(cohesion, cohesion+1, cohesion+2, flockData+j, flockData+j+flockDim, flockData+j+2*flockDim, cohesion);
						count++;
				}
    }

    if(count != 0){
        vector3Mul(cohesion, 1.0/count, cohesion);
        vector3Sub(cohesion, cohesion+1, cohesion+2, flockData+i, flockData+i+flockDim, flockData+i+2*flockDim, cohesion);
    }

    vector3Normalize(cohesion);
    vector3Mul(cohesion, cohesionWeigth, cohesion);
}

void getAlignDirection(int i, float* align){
    
    for(int j = 0; j < flockDim; j++){
				if(neighborhoods[i*flockDim+j]){
        		vector3Sum(align, align+1, align+2, flockData+j+3*flockDim, flockData+j+4*flockDim, flockData+j+5*flockDim, align);
				}
    }

    vector3Normalize(align);
    vector3Mul(align, alignWeigth, align);
}

void moveBoid(int i, float time) { 
    
    for(int j = 0; j < 3; j++){
        flockData[i+j*flockDim] += flockData[i+j*flockDim] * velocity * time;
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
			uint pos = tid - threadsNumDev;
			//myGenerations = myGenerations/2;
			for(uint i = 0; i < myGenerations; i++){
					
					d1 = (curand_uniform(&localState) * minMaxDiffDev + minRandDev) / divDev;
					d2 = (curand_uniform(&localState) * minMaxDiffDev + minRandDev) / divDev;
					d3 = (curand_uniform(&localState) * minMaxDiffDev + minRandDev) / divDev;

					value = sqrt(d1*d1 + d2*d2 + d3*d3);

					//if the magnitude is 0 avoid dividing for 0 and set the value to 0, otherwise calculate the value to normalize
					value = !(value == 0) * 1/(value + (value == 0));

					pos += (i <= lastThreadGenerationsDev) * threadsNumDev + (i > lastThreadGenerationsDev) * (threadsNumDev - 1);
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

		if(tid == 0){
				printf("aaaa: %.5f, %.5f", neighDimDev, toleranceDev);
		}

		if(tid < threadsNumDev){
				
				bool value;
				for(int i = 0; i < flockDimDev; i++){

						if(tid == i){
								neighborhoods[i*flockDimDev+tid] = 0;
						}
						else{
								//value = fabs(sqrt(pow(flockData[tid] - flockData[i], 2) + pow(flockData[tid+flockDimDev] - flockData[i+flockDimDev], 2) + pow(flockData[tid+2*flockDimDev] - flockData[i+2*flockDimDev], 2)) - neighDimDev) < toleranceDev;
								value = sqrt(pow(flockData[tid] - flockData[i], 2) + pow(flockData[tid+flockDimDev] - flockData[i+flockDimDev], 2) + pow(flockData[tid+2*flockDimDev] - flockData[i+2*flockDimDev], 2)) <= neighDimDev;
								neighborhoods[i*flockDimDev+tid] = value;
						}
				}
				
				/*
				uint i = (tid/flockDimDev) * flockDimDev/2;
				uint max = i + flockDimDev/2;
				tid = (tid>=flockDimDev) * (tid-flockDimDev) + (tid<flockDimDev) * tid;

				bool value;
				for(; i < max; i++){

						if(tid == i){
								neighborhoods[i*flockDimDev+tid] = 0;
						}
						else{
								value = sqrt(pow(flockData[tid] - flockData[i], 2) + pow(flockData[tid+flockDimDev] - flockData[i+flockDimDev], 2) + pow(flockData[tid+2*flockDimDev] - flockData[i+2*flockDimDev], 2)) <= neighDimDev;
								neighborhoods[i*flockDimDev+tid] = value;
						}
				}
				*/

				/*
				value = sqrt(pow(flockData[tid] - flockData[i], 2) + pow(flockData[tid+flockDimDev] - flockData[i+flockDimDev], 2) + pow(flockData[tid+2*flockDimDev] - flockData[i+2*flockDimDev], 2)) <= neighDimDev;
				//value = (tid == i) * 0 + !(tid == i) * value;
				neighborhoods[i*flockDimDev+tid] = value;
				//neighborhoods[tid*flockDimDev+i] = value;
				*/
		}
}

/*
 * Kernel for flock update: firstly determines the new direction of each boid based on its neighbors status and secondly its new position moving in the new direction at the given velocity 
 */
__global__ void updateFlock(float* flockData, bool* neighborhoods) {

		uint tid = threadIdx.x + blockDim.x * blockIdx.x;

		if(tid < threadsNumDev){
				
				
		}
}

int main(void) {

		device_name();

		// create events to measure time
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		float milliseconds = 0;
		double cpuTimeStart;
		double cpuTime;

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
		cudaMemcpyToSymbol(toleranceDev, &tolerance, sizeof(toleranceDev));
		cudaMemcpyToSymbol(minRandDev, &minRand, sizeof(minRandDev));
		cudaMemcpyToSymbol(minMaxDiffDev, &minMaxDiff, sizeof(minMaxDiffDev));
		cudaMemcpyToSymbol(divDev, &div, sizeof(divDev));
		cudaMemcpyToSymbol(threadsNumDev, &threadsNum, sizeof(threadsNumDev));
		cudaMemcpyToSymbol(generationsPerThreadDev, &generationsPerThread, sizeof(generationsPerThreadDev));
		cudaMemcpyToSymbol(lastThreadGenerationsDev, &lastThreadGenerations, sizeof(lastThreadGenerationsDev));

		printf("\nthreadsNum: %i\n", threadsNum);
		printf("effthreadsNum: %i\n", blockSize * gridSize);
		printf("genperthread: %i\n", generationsPerThread);
		printf("lastthreadgen: %i\n", lastThreadGenerations);
		if(threadsNum > blockSize * gridSize){
				std::cout << "\nNot enough threads" << std::endl;
		}

		// generate flock
		printf("\n\nGPU Flock generation...\n");
		cudaEventRecord(start);

		initializeStates<<<gridSize, blockSize>>>(time(NULL), states);

		generateBoidsStatus<<<gridSize, blockSize>>>(flockData, states);

		cudaEventRecord(stop);
		CHECK(cudaEventSynchronize(stop));
		cudaEventElapsedTime(&milliseconds, start, stop);
		printf("    GPU elapsed time: %.5f (sec)\n", milliseconds / 1000);

		//print the first 10 and the last 10 boids to check the generation correctness 
		for(int i = 0; i < 10; i++){
				std::cout << std::setprecision(4) << "Boid " << i << ": " << "pos(" << flockData[i] << ", " << flockData[i+flockDim] << ", " << flockData[i+flockDim*2] << 
					"); dir(" << flockData[i+flockDim*3] << ", " << flockData[i+flockDim*4] << ", " << flockData[i+flockDim*5] << ")" << std::endl;
				printf("Boid direction magnitude: %.5f\n", vector3Magnitude(flockData+i+flockDim*3, flockData+i+flockDim*4, flockData+i+flockDim*5));
		}

		for(int i = 0; i < 10; i++){
				std::cout << std::setprecision(4) << "Boid " << flockDim-1-i << ": " << "pos(" << flockData[flockDim-1-i] << ", " << flockData[flockDim-1-i+flockDim] << ", " << flockData[flockDim-1-i+flockDim*2] << 
					"); dir(" << flockData[flockDim-1-i+flockDim*3] << ", " << flockData[flockDim-1-i+flockDim*4] << ", " << flockData[flockDim-1-i+flockDim*5] << ")" << std::endl;
				printf("Boid direction magnitude: %.5f\n", vector3Magnitude(flockData+flockDim-1-i+flockDim*3, flockData+flockDim-1-i+flockDim*4, flockData+flockDim-1-i+flockDim*5));
		}

		for(int i = flockDim/2-5; i < flockDim/2+5; i++){
				std::cout << std::setprecision(4) << "Boid " << i << ": " << "pos(" << flockData[i] << ", " << flockData[i+flockDim] << ", " << flockData[i+flockDim*2] << 
					"); dir(" << flockData[i+flockDim*3] << ", " << flockData[i+flockDim*4] << ", " << flockData[i+flockDim*5] << ")" << std::endl;
				printf("Boid direction magnitude: %.5f\n", vector3Magnitude(flockData+i+flockDim*3, flockData+i+flockDim*4, flockData+i+flockDim*5));
		}

		// generate flock sequentially to measure the speed-up
		/*
		float* flockDataSeq = (float*) malloc(numsToGenerate * sizeof(float));

		printf("\n\nCPU Flock generation...\n");
		cpuTimeStart = seconds();

		for(int i = 0; i < numsToGenerate; i+=6){
			flockDataSeq[i] = (minRand + rand() % (maxRand + 1 - minRand)) / div;
			flockDataSeq[i+1] = (minRand + rand() % (maxRand + 1 - minRand)) / div;
			flockDataSeq[i+2] = (minRand + rand() % (maxRand + 1 - minRand)) / div;
			flockDataSeq[i+3] = (minRand + rand() % (maxRand + 1 - minRand)) / div;
			flockDataSeq[i+4] = (minRand + rand() % (maxRand + 1 - minRand)) / div;
			flockDataSeq[i+5] = (minRand + rand() % (maxRand + 1 - minRand)) / div;
		
			vector3Normalize(flockDataSeq+i+3);
		}

		cpuTime = seconds() - cpuTimeStart;
		printf("    CPU elapsed time: %.5f (sec)\n", cpuTime);

		printf("				Speedup: %.2f\n", cpuTime/(milliseconds / 1000));

		free(flockDataSeq);
		*/

		CHECK(cudaFree(states));

		// prepare for neighborhood calculation
		// neighborhoods data stored in a boolean matrix
		CHECK(cudaMallocManaged((void **) &neighborhoods, flockDim * flockDim * sizeof(bool)));
		threadsNum = flockDim; //flockDim*2; 
		int neighBlockSize = NEIGH_BLOCK_SIZE;
		int neighGridSize = (threadsNum + blockSize - 1)/blockSize;

		printf("\nthreadsNum: %i\n", threadsNum);
		printf("gridSize: %i\n", neighGridSize);
		printf("effthreadsNum: %i\n", neighBlockSize * neighGridSize);
		if(threadsNum > neighBlockSize * neighGridSize){
				std::cout << "\nNot enough threads" << std::endl;
		}

		// update total threads number in constant memory
		cudaMemcpyToSymbol(threadsNumDev, &threadsNum, sizeof(threadsNumDev));

		// compute all neighborhoods
		printf("\n\nGPU Neighborhoods computation...\n");
		cudaEventRecord(start);

		computeAllNeighborhoods<<<neighGridSize, neighBlockSize>>>(flockData, neighborhoods);

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

		/*
		for(int i = 0; i < flockDim; i++){
				std::cout << std::setprecision(4) << "Boid " << i << ": " << "pos(" << flockData[i] << ", " << flockData[i+flockDim] << ", " << flockData[i+flockDim*2] << 
				"); dir(" << flockData[i+flockDim*3] << ", " << flockData[i+flockDim*4] << ", " << flockData[i+flockDim*5] << ")" << std::endl;
    }
		*/

		//printFlock();		
		//std::cout << std::endl;
		//printNeighborhoods();
		//std::cout << std::endl;
		//std::cout << std::endl;

		// calculate neighborhoods sequentially to check the result and measure the speed-up
		
		neighborhoodsSeq = (bool*) malloc(flockDim * flockDim * sizeof(bool));

		printf("\n\nCPU Neighborhoods computation...\n");
		cpuTimeStart = seconds();

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

		cpuTime = seconds() - cpuTimeStart;
		printf("    CPU elapsed time: %.5f (sec)\n", cpuTime);

		printf("				Speedup: %.2f\n", cpuTime/(milliseconds / 1000));
		

		
		bool correct = 1;
		for(int i = 0; i < flockDim; i++){
				
				if(!correct){
						break;
				}

				for(int j = 0; j < flockDim; j++){
						if(neighborhoodsSeq[i*flockDim+j] != neighborhoods[i*flockDim+j] && vector3Distance(flockData+i, flockData+i+flockDim, flockData+i+2*flockDim, flockData+j, flockData+j+flockDim, flockData+j+2*flockDim) != neighDim){
								correct = 0;
								printf("seq: %i, %.5f --- par: %i, %.5f\n", neighborhoodsSeq[i*flockDim+j], vector3Distance(flockData+i, flockData+i+flockDim, flockData+i+2*flockDim, flockData+j, flockData+j+flockDim, flockData+j+2*flockDim),
								        neighborhoods[i*flockDim+j], sqrt(pow(flockData[i] - flockData[j], 2) + pow(flockData[i+flockDim] - flockData[j+flockDim], 2) + pow(flockData[i+2*flockDim] - flockData[j+2*flockDim], 2)));
								printf("i: %i --- j: %i\n", i,j);
								
								std::cout << std::setprecision(4) << "Boid " << i << ": " << "pos(" << flockData[i] << ", " << flockData[i+flockDim] << ", " << flockData[i+flockDim*2] << 
									"); dir(" << flockData[i+flockDim*3] << ", " << flockData[i+flockDim*4] << ", " << flockData[i+flockDim*5] << ")" << std::endl;
								printf("Boid direction magnitude: %.5f\n", vector3Magnitude(flockData+i+flockDim*3, flockData+i+flockDim*4, flockData+i+flockDim*5));

								std::cout << std::setprecision(4) << "Boid " << j << ": " << "pos(" << flockData[j] << ", " << flockData[j+flockDim] << ", " << flockData[j+flockDim*2] << 
									"); dir(" << flockData[j+flockDim*3] << ", " << flockData[j+flockDim*4] << ", " << flockData[j+flockDim*5] << ")" << std::endl;
								printf("Boid direction magnitude: %.5f\n", vector3Magnitude(flockData+j+flockDim*3, flockData+j+flockDim*4, flockData+j+flockDim*5));
								break;
						}
				}
    }

		printf("\nNeighborhoods computation correctness: %i\n\n", correct);
/*
		for(int i = 0; i < flockDim; i++){
				std::cout << i << ": ";
				for(int j = 0; j < flockDim; j++){
						if(neighborhoods[i*flockDim+j]){
								std::cout << j << ", ";
						}
				}
				std::cout << std::endl;
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
*/
		//prepare for flock updates
		threadsNum = flockDim*2;
		blockSize = UPDATE_BLOCK_SIZE;
		gridSize = (threadsNum + blockSize - 1)/blockSize;

		// start simulation loop that updates the flock each updateTime
		double loopStart = seconds();
		double tmpTime = updateTime;
		while(iterations > 0){

				auto duration = seconds() - loopStart;
				if(duration >= tmpTime)
				{
						// update the flock
						printf("\n\nGPU Flock update...\n");
						cudaEventRecord(start);

						updateFlock<<<gridSize, blockSize>>>(flockData, neighborhoods);

						cudaEventRecord(stop);
						CHECK(cudaEventSynchronize(stop));
						cudaEventElapsedTime(&milliseconds, start, stop);
						printf("    GPU elapsed time: %.5f (sec)\n", milliseconds / 1000);
					
						// update the flock sequentially to check the result and measure the speed-up
						

						printf("\n\nCPU Flock update...\n");
						double cpuTimeStart = seconds();

						updateFlock(updateTime);
					
						double cpuTime = seconds() - cpuTimeStart;
						printf("    CPU elapsed time: %.5f (sec)\n", cpuTime);
					
						printf("				Speedup: %.2f\n", cpuTime/(milliseconds / 1000));
						
						//-------
						
					
						computeAllNeighborhoods<<<neighGridSize, neighBlockSize>>>(flockData, neighborhoods);
					
						tmpTime += updateTime;
						iterations--;

						//std::cout << std::endl;
						//printFlock();
					
						//std::cout << std::endl;
						//printNeighborhoods();
				}
		}

		CHECK(cudaFree(flockData));
		CHECK(cudaFree(neighborhoods));
		//free(neighborhoodsSeq);
		
		cudaDeviceReset();
		return 0;
}