
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

void outputCardInfo();
void Mod4();
void speedTest(int *a, int *b, int *c, int *deva,int *devb,int *devc, int bytes);

int main(int argc,char* argv[]) {
	outputCardInfo();
	Mod4();

	return 0;

}

void outputCardInfo() {
	cudaDeviceProp Card;
	int count;
	cudaGetDeviceCount(&count);
	printf("Number of cards:%i\n", count);
	cudaGetDeviceProperties(&Card, 0);
	std::cout << Card.name << std::endl;;
	std::cout << "MaxThreads: " << Card.maxThreadsPerBlock << " Major: " << Card.major << " Minor: " << Card.minor << " clock Rate: " << Card.clockRate << std::endl;
	std::cout << "total Mem: " << Card.totalGlobalMem << " Total const Mem: " << Card.totalConstMem << " Texture alignment:" << Card.textureAlignment << std::endl;
	std::cout << "Multi processor count	:" << Card.multiProcessorCount <<" Max Threads" << Card.maxThreadsPerBlock;
	printf("\n\n");
}

void Mod4() {
	
	const unsigned int bytes = N * sizeof(int);

	//Host located pageable
	int a[N], b[N], c[N];
	
	//To be pinned memory 
	int *h_pa, *h_pb, *h_pc;

	//Device global mem holders
	int *dev_a, *dev_b, *dev_c;

	//Allocate devices
	cudaMalloc((void**)&dev_a, bytes);
	cudaMalloc((void**)&dev_b, bytes);
	cudaMalloc((void**)&dev_c, bytes);

	//Allocate pinned
	cudaMallocHost((void**)&h_pa, bytes);
	cudaMallocHost((void**)&h_pb, bytes);
	cudaMallocHost((void**)&h_pc, bytes);

	//Populate our arrays with numbers.
	for (int i = 0; i < N; i++) {
		a[i] = -i;
		b[i] = i*i;
		h_pa[i] = -i;
		h_pb[i] = i*i;
	}

	speedTest(a,b, c, dev_a, dev_b, dev_c,bytes);
	speedTest(h_pa, h_pb, h_pc, dev_a, dev_b, dev_c,bytes);
	
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
	cudaFreeHost(h_pa);
	cudaFreeHost(h_pb);
	cudaFreeHost(h_pc);
}

void speedTest(int *a, int *b, int *c, int *deva, int *devb, int *devc,int bytes) {
	cudaMemcpy(deva, a, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(devb, b, bytes, cudaMemcpyHostToDevice);
	auto start = std::chrono::high_resolution_clock::now();
	add << <16, 1024 >> > (deva, devb, devc);
	auto stop = std::chrono::high_resolution_clock::now();
	cudaMemcpy(c, devc, bytes, cudaMemcpyDeviceToHost);
	printf("Kernel finished in:%d\n", stop - start);
}