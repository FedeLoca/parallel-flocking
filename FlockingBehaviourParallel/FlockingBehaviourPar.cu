#include "FlockingBehaviourPar.h"

#define BLOCK_SIZE 512
#define GRID_SIZE 64
//128
//32

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
__global__ void generateBoidsStatus(int minRand, int maxRand, int div, uint flockDim, uint threadsNum, uint generationsPerThread, float *generated, curandState *states) {

	uint tid = threadIdx.x + blockDim.x * blockIdx.x;

	if(tid < threadsNum){
			
			curandState localState = states[tid];

			//if the thread is the last make it generate the numbers for the remaining boids, otherwise make it generate generationsPerThread numbers
			uint myGenerations = (tid == (threadsNum - 1)) * (flockDim - generationsPerThread * tid) + !(tid == (threadsNum - 1)) * generationsPerThread;
	
			float d1, d2, d3;
			float magnitude;
			float value;
			uint pos = tid * generationsPerThread * 6;
			for(uint i = 0; i < myGenerations; i++){
					d1 = (curand_uniform(&localState) * (maxRand - minRand) + minRand) / div;
					d2 = (curand_uniform(&localState) * (maxRand - minRand) + minRand) / div;
					d3 = (curand_uniform(&localState) * (maxRand - minRand) + minRand) / div;

					magnitude = sqrt(d1*d1 + d2*d2 + d3*d3);

					//if the magnitude is 0 avoid dividing for 0 and set the value to 0, otherwise calculate the value to normalize
					value = !(magnitude == 0) * 1/(magnitude + (magnitude == 0));

					generated[pos] = (curand_uniform(&localState) * (maxRand - minRand) + minRand) / div;
					generated[pos + 1] = (curand_uniform(&localState) * (maxRand - minRand) + minRand) / div;
					generated[pos + 2] = (curand_uniform(&localState) * (maxRand - minRand) + minRand) / div;
					generated[pos + 3] = d1 * value;
					generated[pos + 4] = d2 * value;
					generated[pos + 5] = d3 * value;
					pos = pos + 6;
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
	float neighDim = 750000000; // 75000000 dimension of the neighborhood in meters
	int minRand = -500000000; // -50000000 minimum value that can be generated for initial position and direction
	int maxRand = 500000000; // 50000000 maximum value that can be generated for initial position and direction
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

	float *generatedNumsHost = (float *) malloc(numsToGenerate * sizeof(float));
	float *generatedNumsDev;
	curandState *states;
	CHECK(cudaMalloc((void **) &generatedNumsDev, numsToGenerate * sizeof(float)));
	int blockSize = BLOCK_SIZE;
	int gridSize = GRID_SIZE; 
	int generationsPerThread = 5000;  //2500 (flockDim + (blockSize * gridSize) - 1) / (blockSize * gridSize);
	int threadsNum = (flockDim + generationsPerThread - 1) / generationsPerThread;

	if(threadsNum > blockSize * gridSize){
			std::cout << "Not enough threads" << std::endl;
	}

	CHECK(cudaMalloc((void **) &states, blockSize * gridSize * sizeof(curandState)));
	cudaEventRecord(start);

	// generate flock
	printf("\n\nGPU Flock generation...\n");

	initializeStates<<<gridSize, blockSize>>>(time(NULL), states);

  generateBoidsStatus<<<gridSize, blockSize>>>(minRand, maxRand, div, flockDim, threadsNum, generationsPerThread, generatedNumsDev, states);

	cudaEventRecord(stop);
	CHECK(cudaEventSynchronize(stop));
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("    GPU elapsed time: %.5f (sec)\n", milliseconds / 1000);

	CHECK(cudaMemcpy(generatedNumsHost, generatedNumsDev, numsToGenerate * sizeof(float), cudaMemcpyDeviceToHost));

	printf("threadsNum: %i\n", threadsNum);
	printf("effthreadsNum: %i\n", blockSize * gridSize);
  printf("genperthread: %i\n", generationsPerThread);

	for(int i = 0; i < 12; i++){
			printf("first: %.5f\n", generatedNumsHost[i]);
	}
	for(int i = 0; i < 12; i++){
			printf("last: %.5f\n", generatedNumsHost[numsToGenerate - i - 1]);
	}

	std::vector<float> dir;
	float p1, p2, p3, d1, d2, d3;
	for(int i = 0; i < numsToGenerate; i+=6){
			p1 = generatedNumsHost[i];
			p2 = generatedNumsHost[i + 1];
			p3 = generatedNumsHost[i + 2];
			d1 = generatedNumsHost[i + 3];
			d2 = generatedNumsHost[i + 4];
			d3 = generatedNumsHost[i + 5];
			f.addBoid(Boid{i/6, velocity, std::vector<float>{p1,p2,p3}, std::vector<float>{d1,d2,d3}});
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