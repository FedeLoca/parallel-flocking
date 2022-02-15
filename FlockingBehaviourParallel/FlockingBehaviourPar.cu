#include "FlockingBehaviourPar.h"

#define GEN_BLOCK_SIZE 128
#define GEN_GRID_SIZE 32
//384  128
//64   32
#define NEIGH_BLOCK_SIZE 256
//128
#define UPDATE_BLOCK_SIZE 512

float velocity = 20; // boid velocity in meters per second
double updateTime = 1.5; // update time of the simulation in seconds
float separationWeight = 1; // weight of the separation component in the blending
float cohesionWeight = 1; // weight of the cohesion component in the blending
float alignWeight = 1; // weight of the align component in the blending
int flockDim = 1024; // 10000000 number of boids in the flock
float neighDim = 50; // 50 dimension of the neighborhood in meters
float tolerance = 0.1f; //tolerance for float comparison
int minRand = -50000; // -50000 minimum value that can be generated for initial position and direction
int maxRand = 50000; // 50000 maximum value that can be generated for initial position and direction
float decimals = 3; // 3 number of decimal digits in the generated values for initial position and direction
int iterations = 1; // number of updates
int generationsPerThread = 250; //2500 number of boids a thread must generate

float* flockData;
bool* neighborhoods;
bool* neighborhoodsSeq;

__constant__ float movementDev;
__constant__ float separationWeightDev;
__constant__ float cohesionWeightDev;
__constant__ float alignWeightDev;
__constant__ int flockDimDev;
__constant__ float neighDimDev;
__constant__ float toleranceDev;
__constant__ int minRandDev;
__constant__ int minMaxDiffDev;
__constant__ float divDev;
__constant__ int threadsNumDev;
__constant__ int generationsPerThreadDev;
__constant__ int lastThreadGenerationsDev;

/*
 * Initialize cuRAND states
 */
__global__ void initializeStates(uint seed, curandState* states) {

		uint tid = threadIdx.x + blockDim.x * blockIdx.x;

		curand_init(seed, tid, 0, &states[tid]);
}


__device__ void generateBoidGPU(uint i, curandState* localState, float* generated, uint pos, float d1, float d2, float d3, float value){
		
		d1 = (curand_uniform(localState) * minMaxDiffDev + minRandDev) / divDev;
		d2 = (curand_uniform(localState) * minMaxDiffDev + minRandDev) / divDev;
		d3 = (curand_uniform(localState) * minMaxDiffDev + minRandDev) / divDev;

		value = sqrt(d1*d1 + d2*d2 + d3*d3);

		//if the magnitude is 0 avoid dividing for 0 and set the value to 0, otherwise calculate the value to normalize
		value = !(value == 0) * 1/(value + (value == 0));

		pos += (i <= lastThreadGenerationsDev) * threadsNumDev + (i > lastThreadGenerationsDev) * (threadsNumDev - 1);
		generated[pos] = (curand_uniform(localState) * minMaxDiffDev + minRandDev) / divDev;
		generated[pos + flockDimDev] = (curand_uniform(localState) * minMaxDiffDev + minRandDev) / divDev;
		generated[pos + 2 * flockDimDev] = (curand_uniform(localState) * minMaxDiffDev + minRandDev) / divDev;
		generated[pos + 3 * flockDimDev] = d1 * value;
		generated[pos + 4 * flockDimDev] = d2 * value;
		generated[pos + 5 * flockDimDev] = d3 * value;
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

					//generateBoidGPU(i, &localState, generated, pos, d1, d2, d3, value);
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
		}
}

__device__ void computeCohesionGPU(uint tid, bool* neighborhoods, float* flockData, float* cohesions){
		
		cohesions[tid] = 0;
		cohesions[tid+flockDimDev] = 0;
		cohesions[tid+2*flockDimDev] = 0;

		float count = 0.0;
		for(int i = 0; i < flockDimDev; i++){
				
				if(neighborhoods[i*flockDimDev+tid]){
						
						cohesions[tid] += flockData[i];
						cohesions[tid+flockDimDev] += flockData[i+flockDimDev];
						cohesions[tid+2*flockDimDev] += flockData[i+2*flockDimDev];
						count += 1;
				}
		}

		if(count != 0){
				
				count = 1.0/count;
				cohesions[tid] *= count;
				cohesions[tid+flockDimDev] *= count;
				cohesions[tid+2*flockDimDev] *= count;

				cohesions[tid] -= flockData[tid];
				cohesions[tid+flockDimDev] -= flockData[tid+flockDimDev];
				cohesions[tid+2*flockDimDev] -= flockData[tid+2*flockDimDev];
		}

		float normValue;
		normValue = sqrt(cohesions[tid]*cohesions[tid] + cohesions[tid+flockDimDev]*cohesions[tid+flockDimDev] + cohesions[tid+2*flockDimDev]*cohesions[tid+2*flockDimDev]);
		normValue = !(normValue == 0) * 1/(normValue + (normValue == 0));
		normValue *= cohesionWeightDev;

		cohesions[tid] *= normValue;
		cohesions[tid+flockDimDev] *= normValue;
		cohesions[tid+2*flockDimDev] *= normValue;
}

__device__ void computeSeparationGPU(uint tid, bool* neighborhoods, float* flockData, float* separations){
		
		separations[tid] = 0;
		separations[tid+flockDimDev] = 0;
		separations[tid+2*flockDimDev] = 0;

		float tmp1;
		float tmp2;
		float tmp3;
		float magValue;
		float normValue;
		for(int i = 0; i < flockDimDev; i++){
				
				if(neighborhoods[i*flockDimDev+tid]){
						
						tmp1 = flockData[tid] - flockData[i];
						tmp2 = flockData[tid+flockDimDev] - flockData[i+flockDimDev];
						tmp3 = flockData[tid+2*flockDimDev] - flockData[i+2*flockDimDev];

						normValue = sqrt(tmp1*tmp1 + tmp2*tmp2 + tmp3*tmp3);
						magValue = normValue;
						magValue = 1/(magValue + 0.0001f);
						normValue = !(normValue == 0) * 1/(normValue + (normValue == 0));
						normValue *= magValue;

						tmp1 *= normValue;
						tmp2 *= normValue;
						tmp3 *= normValue;

						separations[tid] += tmp1;
						separations[tid+flockDimDev] += tmp2;
						separations[tid+2*flockDimDev] += tmp3;
				}
		}

		normValue = sqrt(separations[tid]*separations[tid] + separations[tid+flockDimDev]*separations[tid+flockDimDev] + separations[tid+2*flockDimDev]*separations[tid+2*flockDimDev]);
		normValue = !(normValue == 0) * 1/(normValue + (normValue == 0));
		normValue *= separationWeightDev;

		separations[tid] *= normValue;
		separations[tid+flockDimDev] *= normValue;
		separations[tid+2*flockDimDev] *= normValue;
}

__device__ void computeAlignGPU(uint tid, bool* neighborhoods, float* flockData, float* aligns){
		
		aligns[tid] = 0;
		aligns[tid+flockDimDev] = 0;
		aligns[tid+2*flockDimDev] = 0;

		for(int i = 0; i < flockDimDev; i++){
				
				if(neighborhoods[i*flockDimDev+tid]){
						
						aligns[tid] += flockData[i+3*flockDimDev];
						aligns[tid+flockDimDev] += flockData[i+4*flockDimDev];
						aligns[tid+2*flockDimDev] += flockData[i+5*flockDimDev];
				}
		}

		float normValue;
		normValue = sqrt(aligns[tid]*aligns[tid] + aligns[tid+flockDimDev]*aligns[tid+flockDimDev] + aligns[tid+2*flockDimDev]*aligns[tid+2*flockDimDev]);
		normValue = !(normValue == 0) * 1/(normValue + (normValue == 0));
		normValue *= alignWeightDev;

		aligns[tid] *= normValue;
		aligns[tid+flockDimDev] *= normValue;
		aligns[tid+2*flockDimDev] *= normValue;
}

/*
 * Kernel for flock update: firstly determines the new direction of each boid based on its neighbors status and secondly its new position moving in the new direction at the given velocity 
 */
__global__ void updateFlock(float* flockData, bool* neighborhoods) {

		uint tid = threadIdx.x + blockDim.x * blockIdx.x;

		if(tid < threadsNumDev){
				
				extern __shared__ float cohesions[];
				float* separations = cohesions+3*flockDimDev;
				float* aligns = cohesions+6*flockDimDev;
				
				if(tid < flockDimDev-1){
						
						// first block calculates cohesion

						/*
						bool isNeighbor;
						isNeighbor = neighborhoods[i*flockDimDev+tid];
						cohesions[tid] += flockData[i] * isNeighbor;
						cohesions[tid+flockDimDev] += flockData[i+flockDimDev] * isNeighbor;
						cohesions[tid+2*flockDimDev] += flockData[i+2*flockDimDev] * isNeighbor;
						tmp += 1 * isNeighbor;
						*/

						computeCohesionGPU(tid, neighborhoods, flockData, cohesions);
				}
				else if(tid >= flockDimDev-1 && tid < tid < 2*flockDimDev-1){
						
						// second block calculates separation

						computeSeparationGPU(tid, neighborhoods, flockData, separations);
				}
				else{
						
						// third block calculates align

						computeAlignGPU(tid, neighborhoods, flockData, aligns);
			  }

				__syncthreads();

				// blend contributions and move in the resulting direction

				float tmp1;
				float tmp2;
				float tmp3;
				if(tid < flockDimDev-1){
						
						tmp1 = cohesions[tid] + separations[tid] + aligns[tid];
						tmp2 = cohesions[tid+flockDimDev] + separations[tid+flockDimDev] + aligns[tid+flockDimDev];
						tmp3 = cohesions[tid+2*flockDimDev] + separations[tid+2*flockDimDev] + aligns[tid+2*flockDimDev];

						flockData[tid] += tmp1 * movementDev;
						flockData[tid+flockDimDev] += tmp2 * movementDev;
						flockData[tid+2*flockDimDev] += tmp3 * movementDev;

						flockData[tid+3*flockDimDev] = tmp1;
						flockData[tid+4*flockDimDev] = tmp2;
						flockData[tid+5*flockDimDev] = tmp3;

				}
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
		for(int i = flockDim-11; i < flockDim-1; i++){
				printBoid(i, flockData, flockDim);
		}
		for(int i = flockDim/2-5; i < flockDim/2+5; i++){
				printBoid(i, flockData, flockDim);
		}

		// generate flock sequentially to measure the speed-up

		float* flockDataSeq = (float*) malloc(numsToGenerate * sizeof(float));

		printf("\n\nCPU Flock generation...\n");
		cpuTimeStart = seconds();

		generateFlock(flockDataSeq, numsToGenerate, maxRand, minRand, div);

		cpuTime = seconds() - cpuTimeStart;
		printf("    CPU elapsed time: %.5f (sec)\n", cpuTime);

		printf("				Speedup: %.2f\n", cpuTime/(milliseconds / 1000));

		free(flockDataSeq);


		CHECK(cudaFree(states));

		// -------------------------------------NEIGHBORHOODS CALCULATION---------------------------------------------

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
		float movement = velocity * updateTime;

		cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);

		threadsNum = flockDim*3;
		int updateBlockSize = UPDATE_BLOCK_SIZE;
		int updateGridSize = (threadsNum + updateBlockSize - 1)/updateBlockSize;

		//prepare for CPU flock update
		float* cohesion = (float*) malloc(3*sizeof(float));
		float* separation = (float*) malloc(3*sizeof(float));
		float* align = (float*) malloc(3*sizeof(float));
		float* finalDirection = (float*) malloc(3*sizeof(float));

		printf("\nthreadsNum: %i\n", threadsNum);
		printf("gridSize: %i\n", updateGridSize);
		printf("effthreadsNum: %i\n", updateBlockSize * updateGridSize);
		if(threadsNum > updateBlockSize * updateGridSize){
				std::cout << "\nNot enough threads" << std::endl;
		}

		// update total threads number in constant memory
		cudaMemcpyToSymbol(threadsNumDev, &threadsNum, sizeof(threadsNumDev));
		cudaMemcpyToSymbol(movementDev, &movement, sizeof(movementDev));

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

						//3 * 3 * flockDim * sizeof(float)
						updateFlock<<<updateGridSize, updateBlockSize, 3 * 3 * flockDim  * sizeof(float)>>>(flockData, neighborhoods);

						cudaEventRecord(stop);
						CHECK(cudaEventSynchronize(stop));
						cudaEventElapsedTime(&milliseconds, start, stop);
						printf("    GPU elapsed time: %.5f (sec)\n", milliseconds / 1000);
					
						//print some boids to check the update correctness 
						for(int i = 0; i < 10; i++){
								printBoid(i, flockData, flockDim);
						}
						for(int i = flockDim-11; i < flockDim-1; i++){
								printBoid(i, flockData, flockDim);
						}
						for(int i = flockDim/2-5; i < flockDim/2+5; i++){
								printBoid(i, flockData, flockDim);
						}
					
						// update the flock sequentially to check the result and measure the speed-up
						
						printf("\n\nCPU Flock update...\n");
						double cpuTimeStart = seconds();

						updateFlock(updateTime, neighborhoodsSeq, flockData, flockDim, neighDim, separationWeight, cohesionWeight, alignWeight);

						double cpuTime = seconds() - cpuTimeStart;
						printf("    CPU elapsed time: %.5f (sec)\n", cpuTime);
					
						printf("				Speedup: %.2f\n", cpuTime/(milliseconds / 1000));
						
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
		
		cudaDeviceReset();
		return 0;
}