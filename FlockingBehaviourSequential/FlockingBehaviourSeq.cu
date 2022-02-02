#include "FlockingBehaviourSeq.h"

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
	float decimals = 2; // 0 number of decimal digits in the generated values for initial position and direction
	int iterations = 1; // number of updates 

  Flock f{};

	srand (time(NULL));
	int div = pow(10, decimals);

	// generate boids with random initial position and direction and add them to the flock
	printf("\n\nCPU Flock generation...\n");
	double cpuTimeStart = seconds();

	std::vector<float> dir;
	float p1, p2, p3, d1, d2, d3;
	for(int i = 0; i < flockDim; i++){
			p1 = minRand + rand() % (maxRand + 1 - minRand);
			p2 = minRand + rand() % (maxRand + 1 - minRand);
			p3 = minRand + rand() % (maxRand + 1 - minRand);
			d1 = minRand + rand() % (maxRand + 1 - minRand);
			d2 = minRand + rand() % (maxRand + 1 - minRand);
			d3 = minRand + rand() % (maxRand + 1 - minRand);
			dir = std::vector<float>{d1/div, d2/div, d3/div};
			vector3Normalize(dir);
	}

	double cpuTime = seconds() - cpuTimeStart;
	printf("    CPU elapsed time: %.5f (sec)\n", cpuTime);

	for(int i = 0; i < flockDim; i++){
			f.addBoid(Boid{i, velocity, std::vector<float>{p1/div,p2/div,p3/div}, dir});
	}

/*
	std::vector<float> dir;
	float p1, p2, p3, d1, d2, d3;
	for(int i = 0; i < flockDim; i++){
			p1 = minRand + rand() % (maxRand + 1 - minRand);
			p2 = minRand + rand() % (maxRand + 1 - minRand);
			p3 = minRand + rand() % (maxRand + 1 - minRand);
			d1 = minRand + rand() % (maxRand + 1 - minRand);
			d2 = minRand + rand() % (maxRand + 1 - minRand);
			d3 = minRand + rand() % (maxRand + 1 - minRand);
			dir = std::vector<float>{d1/div, d2/div, d3/div};
			vector3Normalize(dir);
			f.addBoid(Boid{i, velocity, std::vector<float>{p1/div,p2/div,p3/div}, dir});
	}
*/
	cpuTime = seconds() - cpuTimeStart;
	printf("    CPU elapsed time: %.5f (sec)\n", cpuTime);

  // set flock parameters
  f.setBlendingWeigths(separationWeigth, cohesionWeigth, alignWeigth);
	f.setNeighborhoodDim(neighDim);

	// generate neighborhoods
	printf("\n\nCPU neighborhoods generation...\n");
	cpuTimeStart = seconds();
	f.updateNeighborhoodMap();
	cpuTime = seconds() - cpuTimeStart;
	printf("    CPU elapsed time: %.5f (sec)\n", cpuTime);

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
				 	printf("\n\nCPU Flock update...\n");
					cpuTimeStart = seconds();
					f.updateFlock(updateTime);
					cpuTime = seconds() - cpuTimeStart;
					printf("    CPU elapsed time: %.5f (sec)", cpuTime);
				
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