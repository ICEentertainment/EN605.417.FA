
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <chrono>

static const int N = 10000;

__global__ void add(int *a, int *b, int *c) {
	int tid = threadIdx.x+blockIdx.x*blockDim.x;
	while(tid<N)
	{
		c[tid] = a[tid] + b[tid];
		tid += blockDim.x*gridDim.x; // We multiply how many threads per block * how many blocks there are and add that index to tid. 
	}
}
void addHost(int *a, int *b, int *c) {
	for(int i =0; i<N; i++){
		c[i] = a[i] + b[i];
	}
}
void outputCardInfo();

int main(int argc,char* argv[]) {
	outputCardInfo();
	int blocks = 3;
	int threads = 64;
	if (argc == 2) {
		blocks = atoi(argv[1]);
		printf("Blocks changed to:%i\n", blocks);
	}
	else if (argc == 3) {
		blocks = atoi(argv[1]);
		threads = atoi(argv[2]);
		printf("Blocks changed to:%i\n", blocks);
		printf("Threads changed to:%i\n", threads);
	}
	int a[N], b[N], c[N];
	int *dev_a, *dev_b, *dev_c;
	cudaMalloc((void**)&dev_a, N * sizeof(int));
	cudaMalloc((void**)&dev_b, N * sizeof(int));
	cudaMalloc((void**)&dev_c, N * sizeof(int));
	
	//Populate our arrays with numbers.
	for (int i = 0; i < N; i++) {
		a[i] = -i;
		b[i] = i*i;
	}

	cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);
	auto start = std::chrono::high_resolution_clock::now();
	add<<<blocks,threads>>> (dev_a, dev_b, dev_c);
	auto stop = std::chrono::high_resolution_clock::now();
	cudaMemcpy(c, dev_c, N*sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
	auto startHost = std::chrono::high_resolution_clock::now();
	addHost(a, b, c);
	auto stopHost = std::chrono::high_resolution_clock::now();
	std::cout <<std::endl<< " Time elapsed GPU = " << std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count() << "ns\n";
	std::cout << " Time elapsed Host = " << std::chrono::duration_cast<std::chrono::nanoseconds>(stopHost - startHost).count() << "ns\n";
	return 0;

}

void outputCardInfo() {
	cudaDeviceProp Card;
	int count;
	cudaGetDeviceCount(&count);
	std::cout << count << std::endl;
	cudaGetDeviceProperties(&Card, 0);
	std::cout << Card.name << std::endl;;
	std::cout << "MaxThreads: " << Card.maxThreadsPerBlock << " Major: " << Card.major << " Minor: " << Card.minor << " clock Rate: " << Card.clockRate << std::endl;
	std::cout << "total Mem: " << Card.totalGlobalMem << " Total const Mem: " << Card.totalConstMem << " Texture alignment:" << Card.textureAlignment << std::endl;
	std::cout << "Multi processor count	:" << Card.multiProcessorCount <<" Max Threads" << Card.maxThreadsPerBlock;
	printf("\n\n");
}
