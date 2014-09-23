#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <iostream>
#include "common.h"


// GPU Kernel to perform vector substraction.
__global__ void VectorSubsKernel(float* ad, float* bd, float* cd, int size)
{
	// Retrieve our coordinates in the block
	int blockId   = blockIdx.y * gridDim.x + blockIdx.x;			 	
	int threadId = blockId * blockDim.x + threadIdx.x; 
	
	if (threadId<size) 
		cd[threadId] = ad[threadId] - bd[threadId];
}


bool subtractVectorGPU( float* a, float* b, float* c, int size )
{
	// Error return value
	cudaError_t status;
	// Number of bytes in the vector.
	int bytes = size * sizeof(float);
	//int max_block_size = 512;
	// Pointers to the device arrays
	float *ad, *bd, *cd;
	// Allocate memory on the device to store each matrix
	cudaMalloc((void**) &ad, bytes);
	cudaMalloc((void**) &bd, bytes);
	cudaMalloc((void**) &cd, bytes);

	// Copy the host input data to the device
	cudaMemcpy(ad, a, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(bd, b, bytes, cudaMemcpyHostToDevice);
/*	
	// Specify the size of the grid and the size of the block
	dim3 dimBlock(size, 1); // Matrix is contained in a block
	dim3 dimGrid(1, 1); // Only using a single grid element today
*/
	dim3 dimBlock(128, 1);
	
	int gridx = 1;
	int gridy = 1;
	if(size/128 < 65536)
		gridx = ceil((float)size/128);
	else{
		gridx = 65535;
		gridy = ceil((float)size/(128*65535));
	}
	dim3 dimGrid(gridx, gridy); // Only using a single grid element today
	// Launch the kernel on a size-by-size block of threads
	VectorSubsKernel<<<dimGrid, dimBlock>>>(ad, bd, cd, size);
	
	// Wait for completion
	cudaThreadSynchronize();
	// Check for errors
	status = cudaGetLastError();
	if (status != cudaSuccess) {
		std::cout << "Kernel failed: " << cudaGetErrorString(status) << 
		std::endl;
		cudaFree(ad);
		cudaFree(bd);
		cudaFree(cd);
		return false;
	}
	// Retrieve the result matrix
	cudaMemcpy(c, cd, bytes, cudaMemcpyDeviceToHost);	
	// Free device memory
	cudaFree(ad);
	cudaFree(bd);
	cudaFree(cd);
	// Success
	return true;
}
