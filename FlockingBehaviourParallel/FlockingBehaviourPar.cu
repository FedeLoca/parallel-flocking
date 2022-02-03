#include "FlockingBehaviourPar.h"

#define BLOCK_SIZE 384
#define GRID_SIZE 64
//512
//64

/*
 * Initialize cuRAND states
 */
__global__ void initializeStates(uint seed, curandState *states) {

	uint tid = threadIdx.x + blockDim.x * blockIdx.x;

	curand_init(seed, tid, 0, &states[tid]);
}

/*
 * Kernel for flock generation: generates boids with random initial position and direction and adds them to the flock
 */
__global__ void generateBoidsStatus(int minRand, uint minMaxDiff, int div, uint flockDim, uint threadsNum, uint generationsPerThread, float *generated, curandState *states) {

	uint tid = threadIdx.x + blockDim.x * blockIdx.x;

	if(tid < threadsNum){
			
			curandState localState = states[tid];

			//to avoid accesses out of bounds
			//if the thread is the last make it generate the numbers for the remaining boids, otherwise make it generate generationsPerThread numbers
			generationsPerThread = (tid == (threadsNum - 1)) * (flockDim - generationsPerThread * tid) + !(tid == (threadsNum - 1)) * generationsPerThread;
	
			float d1, d2, d3;
			float value;
			uint pos;
			//generationsPerThread = generationsPerThread/2;
			for(uint i = 0; i < generationsPerThread; i++){
					d1 = (curand_uniform(&localState) * minMaxDiff + minRand) / div;
					d2 = (curand_uniform(&localState) * minMaxDiff + minRand) / div;
					d3 = (curand_uniform(&localState) * minMaxDiff + minRand) / div;

					value = sqrt(d1*d1 + d2*d2 + d3*d3);

					//if the magnitude is 0 avoid dividing for 0 and set the value to 0, otherwise calculate the value to normalize
					value = !(value == 0) * 1/(value + (value == 0));

					pos = tid + i * threadsNum;
					generated[pos] = (curand_uniform(&localState) * minMaxDiff + minRand) / div;
					generated[pos + flockDim] = (curand_uniform(&localState) * minMaxDiff + minRand) / div;
					generated[pos + 2 * flockDim] = (curand_uniform(&localState) * minMaxDiff + minRand) / div;
					generated[pos + 3 * flockDim] = d1 * value;
					generated[pos + 4 * flockDim] = d2 * value;
					generated[pos + 5 * flockDim] = d3 * value;
					
					/*
					d1 = (curand_uniform(&localState) * minMaxDiff + minRand) / div;
					d2 = (curand_uniform(&localState) * minMaxDiff + minRand) / div;
					d3 = (curand_uniform(&localState) * minMaxDiff + minRand) / div;

					magnitude = sqrt(d1*d1 + d2*d2 + d3*d3);

					//if the magnitude is 0 avoid dividing for 0 and set the value to 0, otherwise calculate the value to normalize
					value = !(magnitude == 0) * 1/(magnitude + (magnitude == 0));

					pos = tid + (i+generationsPerThread) * threadsNum;
					generated[pos] = (curand_uniform(&localState) * minMaxDiff + minRand) / div;
					generated[pos + flockDim] = (curand_uniform(&localState) * minMaxDiff + minRand) / div;
					generated[pos + 2 * flockDim] = (curand_uniform(&localState) * minMaxDiff + minRand) / div;
					generated[pos + 3 * flockDim] = d1 * value;
					generated[pos + 4 * flockDim] = d2 * value;
					generated[pos + 5 * flockDim] = d3 * value;
					*/
			}		
			
			states[tid] = localState;
	}
}

int main(void) {

  float velocity = 20; // boid velocity in meters per second
	double updateTime = 1.5; // update time of the simulation in seconds
	float separationWeigth = 1; // weight of the separation component in the blending
	float cohesionWeigth = 1; // weight of the cohesion component in the blending
	float alignWeigth = 1; // weight of the align component in the blending
	int flockDim = 100000000; // 10000000 number of boids in the flock
	float neighDim = 75000000; // 75000000 dimension of the neighborhood in meters
	int minRand = -50000000; // -50000000 minimum value that can be generated for initial position and direction
	int maxRand = 50000000; // 50000000 maximum value that can be generated for initial position and direction
	float decimals = 3; // 0 number of decimal digits in the generated values for initial position and direction
	int iterations = 1; // number of updates 

	device_name();

	// events to measure time
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float milliseconds = 0;

	// prepare for flock generation
  Flock f{};
	int div = pow(10, decimals);
	int numsToGenerate = flockDim * 6;

	float *generatedNums;
	curandState *states;
	CHECK(cudaMallocManaged((void **) &generatedNums, numsToGenerate * sizeof(float)));
	int blockSize = BLOCK_SIZE;
	int gridSize = GRID_SIZE; 
	int generationsPerThread = 5000; 
	int threadsNum = (flockDim + generationsPerThread - 1) / generationsPerThread;

	if(threadsNum > blockSize * gridSize){
			std::cout << "Not enough threads" << std::endl;
	}

	CHECK(cudaMalloc((void **) &states, blockSize * gridSize * sizeof(curandState)));
	cudaEventRecord(start);

	// generate flock
	printf("\n\nGPU Flock generation...\n");

	initializeStates<<<gridSize, blockSize>>>(time(NULL), states);

  generateBoidsStatus<<<gridSize, blockSize>>>(minRand, maxRand - minRand, div, flockDim, threadsNum, generationsPerThread, generatedNums, states);

	cudaEventRecord(stop);
	CHECK(cudaEventSynchronize(stop));
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("    GPU elapsed time: %.5f (sec)\n", milliseconds / 1000);

	printf("threadsNum: %i\n", threadsNum);
	printf("effthreadsNum: %i\n", blockSize * gridSize);
  printf("genperthread: %i\n", generationsPerThread);

	for(int i = 0; i < 12; i++){
			printf("first: %.5f\n", generatedNums[i]);
	}
	for(int i = 0; i < 12; i++){
			printf("last: %.5f\n", generatedNums[numsToGenerate - i - 1]);
	}

	std::vector<float> dir;
	float p1, p2, p3, d1, d2, d3;
	uint threadGenerations;
	bool a = true;
	for(int i = 0; i < threadsNum; i++){

		  if(i == (threadsNum - 1)){
				
				  threadGenerations = flockDim - generationsPerThread * i;
		  }
		  else{
				
				  threadGenerations = generationsPerThread;
		  }
			
			for(int j = 0; j < threadGenerations; j++){
					
					p1 = generatedNums[i + j * threadsNum];
					p2 = generatedNums[i + j * threadsNum + flockDim];
					p3 = generatedNums[i + j * threadsNum + 2 * flockDim];
					d1 = generatedNums[i + j * threadsNum + 3 * flockDim];
					d2 = generatedNums[i + j * threadsNum + 4 * flockDim];
					d3 = generatedNums[i + j * threadsNum + 5 * flockDim];
					f.addBoid(Boid{i/6, velocity, std::vector<float>{p1,p2,p3}, std::vector<float>{d1,d2,d3}});

					if(a){
							
							printf("p1: %.5f\n", p1);
							printf("p2: %.5f\n", p2);
							printf("p3: %.5f\n", p3);
							printf("d1: %.5f\n", d1);
							printf("d2: %.5f\n", d2);
							printf("d3: %.5f\n", d3);
							a = false;
					}
			}
	}

	cudaEventRecord(stop);
	CHECK(cudaEventSynchronize(stop));
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("    GPU elapsed time: %.5f (sec)\n", milliseconds / 1000);

	CHECK(cudaFree(generatedNumsDev));
	CHECK(cudaFree(states));

  // set flock parameters
	f.setBlendingWeigths(separationWeigth, cohesionWeigth, alignWeigth);
	f.setNeighborhoodDim(neighDim);

  // generate neighborhoods
	printf("\n\nCPU neighborhoods generation...\n");
	double gpuTimeStart = seconds();
	f.updateNeighborhoodMap();
	double gpuTime = seconds() - gpuTimeStart;
	printf("    CPU elapsed time: %.5f (sec)\n", gpuTime);

	//f.print();
	//std::cout << std::endl;
	//f.printNeighborhoods();

	// start simulation loop that updates the flock each updateTime
	double loopStart = seconds();
	double tmpTime = updateTime;
	while(iterations > 0){

	 		auto duration = seconds() - loopStart;
	 		if(duration >= tmpTime)
			{
				 	printf("\n\nGPU Flock update...\n");
					gpuTimeStart = seconds();
					f.updateFlock(updateTime);
					gpuTime = seconds() - gpuTimeStart;
					printf("    GPU elapsed time: %.5f (sec)", gpuTime);
				
					tmpTime += updateTime;
					iterations--;

					//std::cout << std::endl;
				  //f.print();
				
					//std::cout << std::endl;
					//f.printNeighborhoods();
			}
	}
	
	return 0;
}