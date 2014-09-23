#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <iostream>
#include "common.h"

// GPU Kernel to perform vector scaling.
__global__ void VectorScaleKernel(float* ad, float* cd, float scaleFactord, int size)
{
	// Retrieve our coordinates in the block
	int blockId   = blockIdx.y * gridDim.x + blockIdx.x;			 	
	int threadId = blockId * blockDim.x + threadIdx.x; 
	float scaleFactor = scaleFactord;

	// Perform addition
	if (threadId<size) 
	cd[threadId] = ad[threadId] * scaleFactor;
}

bool scaleVectorGPU( float* a, float* c, float scaleFactor, int size )
{
	// Error return value
	cudaError_t status;
	// Number of bytes in the vector.
	int bytes = size * sizeof(float);
	// Pointers to the device arrays
	float *ad, *cd, scaleFactord;
	// Allocate memory on the device to store each matrix
	cudaMalloc((void**) &ad, bytes);
	cudaMalloc((void**) &cd, bytes);
	//cudaMalloc((void**) &scaleFactord, sizeof(float));
	// Copy the host input data to the device
	cudaMemcpy(ad, a, bytes, cudaMemcpyHostToDevice);
	scaleFactord = scaleFactor;

	//scaleFactord = scaleFactor;

	dim3 dimBlock(128, 1);
	
	int gridx = 1;
	int gridy = 1;
	if(size/128 < 65536)
		gridx = ceil((float)size/128);
	else{
		gridx = 65535;
		gridy = ceil((float)size/(128*65535));
	}
	dim3 dimGrid(gridx, gridy); 
	
	// Launch the kernel on a size-by-size block of threads
	VectorScaleKernel<<<dimGrid, dimBlock>>>(ad, cd, scaleFactord, size);
	
	// Wait for completion
	cudaThreadSynchronize();
	// Check for errors
	status = cudaGetLastError();
	if (status != cudaSuccess) {
		std::cout << "Kernel failed: " << cudaGetErrorString(status) << 
		std::endl;
		cudaFree(ad);
		cudaFree(cd);
		return false;
	}
	// Retrieve the result matrix
	cudaMemcpy(c, cd, bytes, cudaMemcpyDeviceToHost);	
	// Free device memory
	cudaFree(ad);
	cudaFree(cd);
	// Success
	return true;
}
