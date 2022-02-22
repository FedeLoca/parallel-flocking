#include "FlockingBehaviour.h"

#define GEN_BLOCK_SIZE 128
#define GEN_GRID_SIZE 16
//Config: fd->40960  gen->20  b_s->128  g_s->16  -----  gen->1  b_s->128  g_s->320 (higher occupancy, but less speed-up)
//Config: fd->102400  gen->50  b_s->128  g_s->16
//Config: fd->10240000  gen->2500  b_s->128  g_s->32
//Config: fd->102400000  gen->5000  b_s->320  g_s->64  -----  gen->2500  b_s->128  g_s->160 (higher occupancy, but less speed-up)

#define NEIGH_BLOCK_SIZE 1024

#define DIRECTION_BLOCK_SIZE 192 
//must be divisible by 3 nd 32

#define UPDATE_BLOCK_SIZE 128 

float velocity = 20; // boid velocity in meters per second
double updateTime = 1.5; // update time of the simulation in seconds
float separationWeight = 1; // weight of the separation component in the blending
float cohesionWeight = 1; // weight of the cohesion component in the blending
float alignWeight = 1; // weight of the align component in the blending
int flockDim = 40960; // number of boids in the flock
float neighDim = 50; // dimension of the neighborhood in meters
float tolerance = 0.001f; //tolerance for float comparison
int minRand = -50000; // minimum value that can be generated for initial position and direction
int maxRand = 50000; // maximum value that can be generated for initial position and direction
float decimals = 3; // number of decimal digits in the generated values for initial position and direction
int iterations = 1; // number of updates
int generationsPerThread = 20; // number of boids a thread must generate

float* flockData;
float* flockDataSeq;
bool* neighborhoods;
bool* neighborhoodsSeq;
float* tmp;

__constant__ float movementDev;
__constant__ float separationWeightDev;
__constant__ float cohesionWeightDev;
__constant__ float alignWeightDev;
__constant__ int flockDimDev;
__constant__ float neighDimDev;
__constant__ int unitNumDev;
__constant__ float toleranceDev;
__constant__ int minRandDev;
__constant__ int minMaxDiffDev;
__constant__ float divDev;
__constant__ int threadsNumDev;
__constant__ int generationsPerThreadDev;
__constant__ int lastThreadGenerationsDev;

/*
 * Kernel for cuRAND states initialization
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
			int myGenerations;
			if(tid == (threadsNumDev - 1)){
					myGenerations = lastThreadGenerationsDev;
			}
			else{
					myGenerations = generationsPerThreadDev;
			}
	
			float d1, d2, d3;
			float value;
			uint pos = tid - threadsNumDev;
			for(uint i = 0; i < myGenerations; i++){
					
					//generate the boid direction
					d1 = (curand_uniform(&localState) * minMaxDiffDev + minRandDev) / divDev;
					d2 = (curand_uniform(&localState) * minMaxDiffDev + minRandDev) / divDev;
					d3 = (curand_uniform(&localState) * minMaxDiffDev + minRandDev) / divDev;

					value = sqrt(d1*d1 + d2*d2 + d3*d3);

					//if the magnitude of the direction is 0 the value must be 1 to avoid dividing for 0, otherwise calculate the value to normalize it as 1/magnitude
					value = !(value == 0) * 1/(value + (value == 0));

					//if the last thread finished its generations we have to skip one position less
					if(i > lastThreadGenerationsDev){
							pos += (threadsNumDev - 1);
					}
					else{
							pos += threadsNumDev;
					}

					//generate and save the boid position
					generated[pos] = (curand_uniform(&localState) * minMaxDiffDev + minRandDev) / divDev;
					generated[pos + flockDimDev] = (curand_uniform(&localState) * minMaxDiffDev + minRandDev) / divDev;
					generated[pos + 2 * flockDimDev] = (curand_uniform(&localState) * minMaxDiffDev + minRandDev) / divDev;

					//normalize and save boid direction
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

				uint i = (tid/flockDimDev) * flockDimDev/unitNumDev;
				uint max = i + flockDimDev/unitNumDev;
				tid = tid % flockDimDev;

				if(max > flockDimDev){
						printf("error i: %i\n",i);
				}
				if(tid >= flockDimDev){
						printf("error tid: %i\n",tid);
				}

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
		}
}

/*
 * Device function for the computation of the cohesion component of one boid: the component is the normalized vector from the boid current position to the average position of its neighbours
 */
__device__ void computeCohesionGPU(uint boidId, uint sharedOffset, uint unitSize, bool* neighborhoods, float* flockData, float* cohesions){
		
		cohesions[sharedOffset] = 0;
		cohesions[sharedOffset+unitSize] = 0;
		cohesions[sharedOffset+2*unitSize] = 0;

		//compute the sum of all the neighbours positions
		float count = 0.0;
		for(int i = 0; i < flockDimDev; i++){
				
				if(neighborhoods[i*flockDimDev+boidId]){
						
						cohesions[sharedOffset] += flockData[i];
						cohesions[sharedOffset+unitSize] += flockData[i+flockDimDev];
						cohesions[sharedOffset+2*unitSize] += flockData[i+2*flockDimDev];
						count += 1;
				}
		}

		//calculate the average position and the vector to it only if there is at least one neighbours, otherwise the cohesion component remains the zero vector
		if(count != 0){
				
				count = 1.0/count;
				cohesions[sharedOffset] *= count;
				cohesions[sharedOffset+unitSize] *= count;
				cohesions[sharedOffset+2*unitSize] *= count;

				cohesions[sharedOffset] -= flockData[boidId];
				cohesions[sharedOffset+unitSize] -= flockData[boidId+flockDimDev];
				cohesions[sharedOffset+2*unitSize] -= flockData[boidId+2*flockDimDev];
		}

		//normalize and weight the component
		float normValue;
		normValue = sqrt(cohesions[sharedOffset]*cohesions[sharedOffset] + cohesions[sharedOffset+unitSize]*cohesions[sharedOffset+unitSize] + cohesions[sharedOffset+2*unitSize]*cohesions[sharedOffset+2*unitSize]);
		normValue = !(normValue == 0) * 1/(normValue + (normValue == 0));
		normValue *= cohesionWeightDev;

		cohesions[sharedOffset] *= normValue;
		cohesions[sharedOffset+unitSize] *= normValue;
		cohesions[sharedOffset+2*unitSize] *= normValue;
}

/*
 * Device function for the computation of the separation component of one boid: the component is the normalized average repulsion vector from the neighbours
 */
__device__ void computeSeparationGPU(uint boidId, uint sharedOffset, uint unitSize, bool* neighborhoods, float* flockData, float* separations){
		
		separations[sharedOffset] = 0;
		separations[sharedOffset+unitSize] = 0;
		separations[sharedOffset+2*unitSize] = 0;

		//compute the average repulsion vector by summing the repulsion vectors
		float tmp1;
		float tmp2;
		float tmp3;
		float magValue;
		float normValue;
		for(int i = 0; i < flockDimDev; i++){
				
				if(neighborhoods[i*flockDimDev+boidId]){
						
						//calculate the vector from the boid position to the neighbour position
						tmp1 = flockData[boidId] - flockData[i];
						tmp2 = flockData[boidId+flockDimDev] - flockData[i+flockDimDev];
						tmp3 = flockData[boidId+2*flockDimDev] - flockData[i+2*flockDimDev];

						//normalize it and divide it by its magnitude to obtain the repulsion vector
						normValue = sqrt(tmp1*tmp1 + tmp2*tmp2 + tmp3*tmp3);
						magValue = normValue;
						magValue = 1/(magValue + 0.0001);
						normValue = !(normValue == 0) * 1/(normValue + (normValue == 0));
						normValue *= magValue;

						tmp1 *= normValue;
						tmp2 *= normValue;
						tmp3 *= normValue;

						//sum it to the current separation
						separations[sharedOffset] += tmp1;
						separations[sharedOffset+unitSize] += tmp2;
						separations[sharedOffset+2*unitSize] += tmp3;
				}
		}

		//normalize and weight the component
		normValue = sqrt(separations[sharedOffset]*separations[sharedOffset] + separations[sharedOffset+unitSize]*separations[sharedOffset+unitSize] + separations[sharedOffset+2*unitSize]*separations[sharedOffset+2*unitSize]);
		normValue = !(normValue == 0) * 1/(normValue + (normValue == 0));
		normValue *= separationWeightDev;

		separations[sharedOffset] *= normValue;
		separations[sharedOffset+unitSize] *= normValue;
		separations[sharedOffset+2*unitSize] *= normValue;
}

/*
 * Device function for the computation of the align component of one boid: the component is the average direction of the neighbours
 */
__device__ void computeAlignGPU(uint boidId, uint sharedOffset, uint unitSize, bool* neighborhoods, float* flockData, float* aligns){
		
		aligns[sharedOffset] = 0;
		aligns[sharedOffset+unitSize] = 0;
		aligns[sharedOffset+2*unitSize] = 0;

		//compute the average direction by summing the neighbours directions
		for(int i = 0; i < flockDimDev; i++){
				
				if(neighborhoods[i*flockDimDev+boidId]){
						
						aligns[sharedOffset] += flockData[i+3*flockDimDev];
						aligns[sharedOffset+unitSize] += flockData[i+4*flockDimDev];
						aligns[sharedOffset+2*unitSize] += flockData[i+5*flockDimDev];
				}
		}

		//normalize and weight the component
		float normValue;
		normValue = sqrt(aligns[sharedOffset]*aligns[sharedOffset] + aligns[sharedOffset+unitSize]*aligns[sharedOffset+unitSize] + aligns[sharedOffset+2*unitSize]*aligns[sharedOffset+2*unitSize]);
		normValue = !(normValue == 0) * 1/(normValue + (normValue == 0));
		normValue *= alignWeightDev;

		aligns[sharedOffset] *= normValue;
		aligns[sharedOffset+unitSize] *= normValue;
		aligns[sharedOffset+2*unitSize] *= normValue;
}

/*
 * Kernel for computing the new direction: determines the new direction of each boid based on its neighbors status
 */
__global__ void computeDirection(float* flockData, bool* neighborhoods, float* tmp, uint unitSize) {

		uint sharedOffset = threadIdx.x % unitSize;
		uint boidId = unitSize * blockIdx.x + sharedOffset;

		if(boidId < flockDimDev){
				
				extern __shared__ float cohesions[];
				float* separations = cohesions+3*unitSize;
				float* aligns = cohesions+6*unitSize;
				
				if(threadIdx.x < unitSize){
						
						// first unit calculates cohesion

						computeCohesionGPU(boidId, sharedOffset, unitSize, neighborhoods, flockData, cohesions);
				}
				else if(threadIdx.x >= unitSize && threadIdx.x < 2*unitSize){
						
						// second unit calculates separation

						computeSeparationGPU(boidId, sharedOffset, unitSize, neighborhoods, flockData, separations);
				}
				else{
						
						// third unit calculates align

						computeAlignGPU(boidId, sharedOffset, unitSize, neighborhoods, flockData, aligns);
			  }

				__syncthreads();

				// blend contributions and normalize them

				float tmp1;
				float tmp2;
				float tmp3;
				if(threadIdx.x < unitSize){
						
						tmp1 = cohesions[sharedOffset] + separations[sharedOffset] + aligns[sharedOffset];
						tmp2 = cohesions[sharedOffset+unitSize] + separations[sharedOffset+unitSize] + aligns[sharedOffset+unitSize];
						tmp3 = cohesions[sharedOffset+2*unitSize] + separations[sharedOffset+2*unitSize] + aligns[sharedOffset+2*unitSize];

						float normValue;
						normValue = sqrt(tmp1*tmp1 + tmp2*tmp2 + tmp3*tmp3);
						normValue = !(normValue == 0) * 1/(normValue + (normValue == 0));

						tmp1 *= normValue;
						tmp2 *= normValue;
						tmp3 *= normValue;

						tmp[boidId] = tmp1;
						tmp[boidId+flockDimDev] = tmp2;
						tmp[boidId+2*flockDimDev] = tmp3;
				}
		}
}

/*
 * Kernel for flock update: determines each boid new position moving in the new direction at the given velocity 
 */
__global__ void updateFlock(float* flockData, float* tmp) {

		uint tid = threadIdx.x + blockDim.x * blockIdx.x;

		if(tid < threadsNumDev){

				//update the direction only if the new direction is not the zero vector
				if(tmp[tid] != 0 || tmp[tid+flockDimDev] != 0 || tmp[tid+2*flockDimDev] != 0){
						flockData[tid+3*flockDimDev] = tmp[tid];
						flockData[tid+4*flockDimDev] = tmp[tid+flockDimDev];
						flockData[tid+5*flockDimDev] = tmp[tid+2*flockDimDev];
				}

				//move in the saved direction
				flockData[tid] += flockData[tid+3*flockDimDev] * movementDev;
				flockData[tid+flockDimDev] += flockData[tid+4*flockDimDev] * movementDev;
				flockData[tid+2*flockDimDev] += flockData[tid+5*flockDimDev] * movementDev;
		}
}

int main(void) {

		device_name();

		printf("\nFlock dimension: %i\n", flockDim);
		printf("Neighborhood dimension: %.2f\n", neighDim);
		printf("Velocity: %.2f\n", velocity);
		printf("Update time: %.2f\n", updateTime);
		printf("Iterations: %i\n", iterations);
		printf("Separation weight: %.2f\n", separationWeight);
		printf("Cohesion weight: %.2f\n", cohesionWeight);
		printf("Align weight: %.2f\n", alignWeight);
		printf("Minimum random number: %i\n", minRand);
		printf("Maximum random number: %i\n", maxRand);
		printf("Decimal digits of random numbers: %.2f\n", decimals);

		// create events to measure time
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		float milliseconds = 0;
		double cpuTimeStart;
		double cpuTime;

		// -------------------------------------FLOCK GENERATION---------------------------------------------

		// prepare for flock generation
		int minMaxDiff = maxRand - minRand;
		float div = pow(10.0, decimals);
		int numsToGenerate = flockDim * 6;

		cudaFuncSetCacheConfig(generateBoidsStatus, cudaFuncCachePreferL1);
		cudaFuncSetCacheConfig(initializeStates, cudaFuncCachePreferL1);

		float* flockData;
		curandState* states;
		int blockSize = GEN_BLOCK_SIZE;
		int gridSize = GEN_GRID_SIZE; 
		CHECK(cudaMallocManaged((void **) &flockData, numsToGenerate * sizeof(float)));
		CHECK(cudaMalloc((void **) &states, blockSize * gridSize * sizeof(curandState)));
		int threadsNum = (flockDim + generationsPerThread - 1) / generationsPerThread;
		int lastThreadGenerations = flockDim - generationsPerThread * (threadsNum - 1);

		// initialize all constant values
		cudaMemcpyToSymbol(separationWeightDev, &separationWeight, sizeof(separationWeightDev));
		cudaMemcpyToSymbol(cohesionWeightDev, &cohesionWeight, sizeof(cohesionWeightDev));
		cudaMemcpyToSymbol(alignWeightDev, &alignWeight, sizeof(alignWeightDev));
		cudaMemcpyToSymbol(flockDimDev, &flockDim, sizeof(flockDimDev));
		cudaMemcpyToSymbol(neighDimDev, &neighDim, sizeof(neighDimDev));
		cudaMemcpyToSymbol(toleranceDev, &tolerance, sizeof(toleranceDev));
		cudaMemcpyToSymbol(minRandDev, &minRand, sizeof(minRandDev));
		cudaMemcpyToSymbol(minMaxDiffDev, &minMaxDiff, sizeof(minMaxDiffDev));
		cudaMemcpyToSymbol(divDev, &div, sizeof(divDev));
		cudaMemcpyToSymbol(threadsNumDev, &threadsNum, sizeof(threadsNumDev));
		cudaMemcpyToSymbol(generationsPerThreadDev, &generationsPerThread, sizeof(generationsPerThreadDev));
		cudaMemcpyToSymbol(lastThreadGenerationsDev, &lastThreadGenerations, sizeof(lastThreadGenerationsDev));

		printf("\nNeeded threads number: %i\n", threadsNum);
		printf("Threads used: %i\n", blockSize * gridSize);
		printf("Block size: %i\n", blockSize);
		printf("Grid size: %i\n",gridSize);
		printf("Generations per thread: %i\n", generationsPerThread);
		printf("Generations of last thread: %i\n", lastThreadGenerations);
		if(threadsNum > blockSize * gridSize){
				std::cout << "\nNot enough threads" << std::endl;
		}

		// generate flock
		printf("\n\nGPU Flock generation...\n");
		cudaEventRecord(start);

		initializeStates<<<gridSize, blockSize>>>(time(NULL), states);

		CHECK(cudaDeviceSynchronize());

		generateBoidsStatus<<<gridSize, blockSize>>>(flockData, states);

		cudaEventRecord(stop);
		CHECK(cudaEventSynchronize(stop));
		cudaEventElapsedTime(&milliseconds, start, stop);
		printf("    GPU elapsed time: %.5f (sec)\n", milliseconds / 1000);

		//print some boids to check the generation correctness 
		for(int i = 0; i < 10; i++){
				printBoid(i, flockData, flockDim);
		}
		for(int i = flockDim/2-5; i < flockDim/2+5; i++){
				printBoid(i, flockData, flockDim);
		}
		for(int i = flockDim-11; i < flockDim-1; i++){
				printBoid(i, flockData, flockDim);
		}


		// generate flock sequentially to measure the speed-up
		flockDataSeq = (float*) malloc(numsToGenerate * sizeof(float));

		printf("\n\nCPU Flock generation...\n");
		cpuTimeStart = seconds();

		generateFlock(flockDataSeq, numsToGenerate, maxRand, minRand, div);

		cpuTime = seconds() - cpuTimeStart;
		printf("    CPU elapsed time: %.5f (sec)\n", cpuTime);

		printf("				Speedup: %.2f\n", cpuTime/(milliseconds / 1000));
		

		CHECK(cudaFree(states));

		// -------------------------------------NEIGHBORHOODS CALCULATION---------------------------------------------

		// prepare for neighborhood calculation
		// neighborhoods data stored in a boolean matrix
		CHECK(cudaMallocManaged((void **) &neighborhoods, flockDim * flockDim * sizeof(bool)));

		cudaFuncSetCacheConfig(computeAllNeighborhoods, cudaFuncCachePreferL1);

		int unitNum = 16; 
		int neighThreadsNum = flockDim*unitNum; 
		int neighBlockSize = NEIGH_BLOCK_SIZE;
		int neighGridSize = (neighThreadsNum + neighBlockSize - 1)/neighBlockSize;

		printf("\nNeeded threads number: %i\n", neighThreadsNum);
		printf("Threads used: %i\n", neighBlockSize * neighGridSize);
		printf("Block size: %i\n", neighBlockSize);
		printf("Grid size: %i\n", neighGridSize);
		printf("Number of units: %i\n", unitNum);
		if(neighThreadsNum > neighBlockSize * neighGridSize){
				std::cout << "\nNot enough threads" << std::endl;
		}

		cudaMemcpyToSymbol(unitNumDev, &unitNum, sizeof(unitNumDev));

		// update total threads number in constant memory
		cudaMemcpyToSymbol(threadsNumDev, &neighThreadsNum, sizeof(threadsNumDev));

		// compute all neighborhoods
		printf("\n\nGPU Neighborhoods computation...\n");
		cudaEventRecord(start);

		computeAllNeighborhoods<<<neighGridSize, neighBlockSize>>>(flockData, neighborhoods);

		cudaEventRecord(stop);
		CHECK(cudaEventSynchronize(stop));
		cudaEventElapsedTime(&milliseconds, start, stop);
		printf("    GPU elapsed time: %.5f (sec)\n", milliseconds / 1000);

		for(int i = 0; i < 12; i++){
				printf("Element %i of neighborhoods: %d\n", i, neighborhoods[i]);
		}
		for(int i = 0; i < 12; i++){
				printf("Element %i of neighborhoods: %d\n", i, neighborhoods[(flockDim-1)*flockDim+flockDim-1-i]);
		}

		//printFlock(flockData, flockDim);		
		//std::cout << std::endl;
		//printNeighborhoods(neighborhoodsSeq, flockDim);
		//std::cout << std::endl;
		//std::cout << std::endl;


		// calculate neighborhoods sequentially to check the result and measure the speed-up
		
		neighborhoodsSeq = (bool*) malloc(flockDim * flockDim * sizeof(bool));

		printf("\n\nCPU Neighborhoods computation...\n");
		cpuTimeStart = seconds();

		computeNeighborhoods(neighborhoodsSeq, flockData, flockDim, neighDim);

		cpuTime = seconds() - cpuTimeStart;
		printf("    CPU elapsed time: %.5f (sec)\n", cpuTime);

		printf("				Speedup: %.2f\n", cpuTime/(milliseconds / 1000));
		
		printf("\nNeighborhoods computation correctness: %i\n\n", checkNeighborhoodsCorrectness(neighborhoods, neighborhoodsSeq, flockData, flockDim));
		

		//printNeighborhoods(neighborhoods, flockDim);
		//printNeighborhoods(neighborhoodsSeq, flockDim);

		// -------------------------------------FLOCK UPDATE---------------------------------------------

		//prepare for flock updates
		CHECK(cudaMallocManaged((void **) &tmp, flockDim * 3));
		float movement = velocity * updateTime;
		cudaMemcpyToSymbol(movementDev, &movement, sizeof(movementDev));

		//priviledge shared memory if a high amount is needed
		//cudaFuncSetCacheConfig(updateFlock, cudaFuncCachePreferShared);
		//cudaFuncSetCacheConfig(computeDirection, cudaFuncCachePreferShared);

		int dirBlockSize = DIRECTION_BLOCK_SIZE;
		int unitSize = dirBlockSize/3;
		int dirGridSize = (flockDim + unitSize - 1)/unitSize;

		int updateThreadsNum = flockDim;
		int updateBlockSize = UPDATE_BLOCK_SIZE;
		int updateGridSize = (updateThreadsNum + updateBlockSize - 1)/updateBlockSize;

		//prepare for CPU flock update
		float* cohesion = (float*) malloc(3*sizeof(float));
		float* separation = (float*) malloc(3*sizeof(float));
		float* align = (float*) malloc(3*sizeof(float));
		float* finalDirection = (float*) malloc(3*sizeof(float));

		printf("\nNeeded threads number: %i\n", dirBlockSize * dirGridSize);
		printf("Threads used: %i\n", dirBlockSize * dirGridSize);
		printf("Block size: %i\n", dirBlockSize);
		printf("Grid size: %i\n", dirGridSize);
		printf("Unit size: %i\n", dirBlockSize/3);

		printf("\nNeeded threads number: %i\n", updateThreadsNum);
		printf("Threads used: %i\n", updateBlockSize * updateGridSize);
		printf("Block size: %i\n", updateBlockSize);
		printf("Grid size: %i\n", updateGridSize);
		if(updateThreadsNum > updateBlockSize * updateGridSize){
				std::cout << "\nNot enough threads" << std::endl;
		}

		// start simulation loop that updates the flock each updateTime
		double loopStart = seconds();
		double tmpTime = updateTime;
		while(iterations > 0){

				auto duration = seconds() - loopStart;
				if(duration >= tmpTime)
				{		

						// update the flock sequentially to check the result and measure the speed-up
						printf("\n\nCPU Flock update...\n");
						double cpuTimeStart = seconds();

						updateFlock(updateTime, neighborhoods, flockData, flockDataSeq, flockDim, neighDim, separationWeight, cohesionWeight, alignWeight);

						double cpuTime = seconds() - cpuTimeStart;
						printf("    CPU elapsed time: %.5f (sec)\n", cpuTime);
					  
					
						// update total threads number in constant memory
						//cudaMemcpyToSymbol(threadsNumDev, &dirThreadsNum, sizeof(threadsNumDev));
					
						// update the flock
						printf("\n\nGPU Flock update...\n");
						cudaEventRecord(start);

						computeDirection<<<dirGridSize, dirBlockSize, 3 * 3 * unitSize * sizeof(float)>>>(flockData, neighborhoods, tmp, unitSize);
					
						cudaDeviceSynchronize();
					
						// update total threads number in constant memory
						cudaMemcpyToSymbol(threadsNumDev, &updateThreadsNum, sizeof(threadsNumDev));
			
						updateFlock<<<updateGridSize, updateBlockSize>>>(flockData, tmp);

						cudaEventRecord(stop);
						CHECK(cudaEventSynchronize(stop));
						cudaEventElapsedTime(&milliseconds, start, stop);
						printf("    GPU elapsed time: %.5f (sec)\n", milliseconds / 1000);
					
						//print some boids to check the update correctness 
						for(int i = 0; i < 10; i++){
								printBoid(i, flockData, flockDim);
						}
						for(int i = flockDim/2-5; i < flockDim/2+5; i++){
								printBoid(i, flockData, flockDim);
						}
					  for(int i = flockDim-11; i < flockDim-1; i++){
								printBoid(i, flockData, flockDim);
						}
					
						printf("				Speedup: %.2f\n", cpuTime/(milliseconds / 1000));
					
						printf("\nUpdate correctness: %i\n\n", checkUpdateCorrectness(flockData, flockDataSeq, flockDim));
					
						// update total threads number in constant memory
						cudaMemcpyToSymbol(threadsNumDev, &neighThreadsNum, sizeof(threadsNumDev));
						
						computeAllNeighborhoods<<<neighGridSize, neighBlockSize>>>(flockData, neighborhoods);
					
						tmpTime += updateTime;
						iterations--;

						//std::cout << std::endl;
						//printFlock(flockData, flockDim);	
					
						//std::cout << std::endl;
						//printNeighborhoods(neighborhoodsSeq, flockDim);
				}
		}

		cudaFree(flockData);
		cudaFree(neighborhoods);
	  //CHECK(cudaFree(flockData));
		//CHECK(cudaFree(neighborhoods));
		free(neighborhoodsSeq);
		free(flockDataSeq);
		
		cudaDeviceReset();

		printf("\nEnd.\n");

		return 0;
}