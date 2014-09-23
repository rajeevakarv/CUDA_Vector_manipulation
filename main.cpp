#include <ctime> // time(), clock()
#include <cmath> // sqrt()
#include <cstdlib> // malloc(), free() 
#include <iostream> // cout, stream 
#include "common.h"
#include <bitset>


const int ITERS = 2;
const int SIZE = 33554432; //Max
//const int SIZE = 128; //Max

void addVectorCPU( float* a, float* b, float* c, int size ){
	for (int i=0; i<size; i++)
		c[i] = a[i] + b[i];
}

void subtractVectorCPU( float* a, float* b, float* c, int size ){

	for (int i=0; i<size; i++){
		c[i] = a[i] - b[i];
	}
}

void scaleVectorCPU( float* a, float* c, float scaleFactor, int size )
{
	for (int i = 0; i < size; i++) {
		c[i] = a[i] * scaleFactor; 
	}
}

void displayAddResults(float* M, float* N, float* P)
{
	for (int i = SIZE-550; i < SIZE; i++)  
		std::cout << M[i] << " "; 
	std :: cout <<  "\n + \n"; 
	for (int j = SIZE-550; j < SIZE; j++) 
		std::cout << N[j] << "," << j << " "; 
	std :: cout << "\n = \n"; 
	for (int k = SIZE-550; k < SIZE; k++) 
		std::cout << P[k] << ","<< k << " "; 
}

void displaySubsResults(float* M, float* N, float* P)
{
	for (int i = SIZE-550; i < SIZE; i++)  
		std::cout << M[i] << " "; 
	std :: cout <<  "\n - \n"; 
	for (int j = SIZE-550; j < SIZE; j++) 
		std::cout << N[j] << " "; 
	std :: cout << "\n = \n"; 
	for (int k = SIZE-550; k < SIZE; k++) 
		std::cout << P[k] << " "; 
}

void displayScaleResults(float* a, float* ccpu)
{
	for (int i = SIZE-100; i < SIZE; i++)  
		std::cout << a[i] << " "; 
	std :: cout <<  "\n * \n"; 
//	std::cout << scalefactor << " "; 
	std :: cout << "\n = \n"; 
	for (int j = SIZE-100; j < SIZE; j++) 
		std::cout << ccpu[j] << " "; 
}


/* Entry point for the program. Allocates space for two matrices, 
calls a function to multiply them, and displays the results. */
int main() 
{ 
	// Allocate the three arrays of SIZE x SIZE floats.
	// The element i,j is represented by index (i*SIZE + j) 
	float* a = new float[SIZE]; 
	float* b = new float[SIZE]; 
	float* ccpu = new float[SIZE];
	float* cgpu = new float[SIZE];
	float scaleFactor = 20;
	clock_t start, end;
	float tcpu, tgpu;
	float sum, L2norm1, delta, L2norm2, L2norm3;

	// Initialize M and N to random integers 
	for (int i = 0; i < SIZE; i++) {
		//a[i] = (float)(rand() % 10); 
		//b[i] = (float)(rand() % 10); 
		a[i] = ((float) rand()) / (float) 1;
		b[i] = ((float) rand()) / (float) 1;
	} 
	/*
	// Addition of the two vectors 
	std :: cout <<  "Addition of two vectors\n\n";
	addVectorCPU(a, b, ccpu, SIZE); 
		displayAddResults(a, b, ccpu);
	std :: cout <<  "\n\nSubstraction of two vectors\n\n";
	subtractVectorCPU(a, b, ccpu, SIZE);
	displaySubsResults(a, b, ccpu);
	std :: cout <<  "\n\nScaling of vector\n\n";
	scaleVectorCPU( a, ccpu, scaleFactor, SIZE );
	displayScaleResults(a, scaleFactor, ccpu);
	
	*/
/*	std :: cout <<  "Addition of two vectors\n\n";
	bool success = addVectorGPU(a, b, cgpu, SIZE);
	if (!success) {
		std::cout << "\n * Device error! * \n" << std::endl;
		return 1;
	}
	displayAddResults(a, b, cgpu);

	std :: cout <<  "Substraction of two vectors\n\n";
	bool success = subtractVectorGPU(a, b, cgpu, SIZE);
	if (!success) {
		std::cout << "\n * Device error! * \n" << std::endl;
		return 1;
	}
	std :: cout <<  "\n\nSubstraction of two vectors\n\n";
	displaySubsResults(a, b, cgpu);
	
	std :: cout <<  "\n\nScaling of vector\n\n";
	bool success = scaleVectorGPU( a, cgpu, scaleFactor, SIZE);
	if (!success) {
		std::cout << "\n * Device error! * \n" << std::endl;
		return 1;
	}
	
	displayScaleResults(a, cgpu);*/



	std::cout << "Operating on a vector of length " << std::bitset<26>(SIZE) << "\n" << std::endl;
	// Adition of the two vectors on the host
	start = clock();
	for (int i = 0; i < ITERS; i++) {
		addVectorCPU(a, b, ccpu, SIZE);
	}
	end = clock();
	tcpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / ITERS;
	// Display the results
	std::cout.precision(10);
	std::cout << "CPU Addition took " << tcpu << " ms:" << std::endl;


	// Perform one warm-up pass and validate
	bool success = addVectorGPU(a, b, cgpu, SIZE);
	if (!success) {
		std::cout << "\n * Device Error! * \n" << "\n" << std::endl;
		return 1;
	}

	// And now time it
	start = clock();
	for (int i = 0; i < ITERS; i++) {
		addVectorGPU(a, b, cgpu, SIZE);
	}
	end = clock();
	tgpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / ITERS;
	// Display the results
	std::cout << "GPU Addition took " << tgpu << " ms:" << std::endl;

	std::cout << "Addition speedup " << (float)tcpu/tgpu << std::endl;

	// Compare the results for correctness
	sum = 0, delta = 0;
	for (int i = 0; i < SIZE; i++) {
		delta += (ccpu[i] - cgpu[i]) * (ccpu[i] - cgpu[i]);
		sum += (ccpu[i] * cgpu[i]);
	}

	L2norm1 = sqrt(delta / sum);
	std::cout << "Addition error: " << L2norm1 << "\n" <<
	((L2norm1 < 1e-6) ? "Passed" : "Failed\n\n") << "\n\n"<< std::endl;


	//Substraction of the two vectors on host.
	
	start = clock();
	for (int i = 0; i < ITERS; i++) {
		subtractVectorCPU(a, b, ccpu, SIZE);
	}
	end = clock();
	tcpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / ITERS;
	// Display the results
	std::cout.precision(10);
	std::cout << "CPU Substraction took " << tcpu << " ms:" << std::endl;

	// Perform one warm-up pass and validate
	success = subtractVectorGPU(a, b, cgpu, SIZE);
	if (!success) {
		std::cout << "\n * Device Error! * \n" << "\n" << std::endl;
		return 1;
	}

	// And now time it
	start = clock();
	for (int i = 0; i < ITERS; i++) {
		subtractVectorGPU(a, b, cgpu, SIZE);
	}
	end = clock();
	tgpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / ITERS;
	// Display the results
	std::cout << "GPU Substraction took " << tgpu << " ms:" << std::endl;

	std::cout << "Substraction speedup " << (float)tcpu/tgpu << std::endl;

	// Compare the results for correctness
	sum = 0, delta = 0;
	for (int i = 0; i < SIZE; i++) {
		delta += (ccpu[i] - cgpu[i]) * (ccpu[i] - cgpu[i]);
		sum += (ccpu[i] * cgpu[i]);
	}

	L2norm2 = sqrt(delta / sum);
	std::cout << "Addition Error: " << L2norm2 << "\n" <<
	((L2norm2 < 1e-6) ? "Passed" : "Failed\n\n") << "\n\n" << std::endl;
	
	//Scale code for Host.
	start = clock();
	for (int i = 0; i < ITERS; i++) {
		scaleVectorCPU( a, ccpu, scaleFactor, SIZE );
	}
	end = clock();
	tcpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / ITERS;
	// Display the results
	std::cout << "CPU Scale function took " << tcpu << " ms:" << std::endl;

		// Perform one warm-up pass and validate
	success = subtractVectorGPU(a, b, cgpu, SIZE);
	if (!success) {
		std::cout << "\n * Device Error! * \n" << "\n" << std::endl;
		return 1;
	}

	// And now time it
	start = clock();
	for (int i = 0; i < ITERS; i++) {
		scaleVectorGPU( a, cgpu, scaleFactor, SIZE );
	}
	end = clock();
	tgpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / ITERS;
	// Display the results
	std::cout << "GPU Scaling took " << tgpu << " ms:" << std::endl;

	std::cout << "Scaling speedup " << (float)tcpu/tgpu << std::endl;

	// Compare the results for correctness
	sum = 0, delta = 0;
	for (int i = 0; i < SIZE; i++) {
		delta += (ccpu[i] - cgpu[i]) * (ccpu[i] - cgpu[i]);
		sum += (ccpu[i] * cgpu[i]);
	}

	L2norm3 = sqrt(delta / sum);
	std::cout << "Scaleing Error: " << L2norm3 << "\n" <<
	((L2norm3 < 1e-6) ? "Passed" : "Failed") << "\n\n"<< std::endl;


	// Release the matrices 
	delete[] a; 
	delete[] b;
	delete[] ccpu; 
	delete[] cgpu;
	getchar();
	return 0;
} 


/*
int main()
{
	// Number of bytes in the matrix.
	int bytes = SIZE * sizeof(float);
	// Timing data
	float tcpu, tgpu;
	clock_t start, end;
	// Allocate the three arrays of SIZE x SIZE floats.
	// The element i,j is represented by index (i*SIZE + j)
	float* a = new float[SIZE];
	float* b = new float[SIZE];
	float* ccpu = new float[SIZE];
	float* cgpu = new float[SIZE];
	
	// Initialize M and N to random integers
	for (int i = 0; i < SIZE; i++) {
		a[i] = (float)(rand() % 10);
		b[i] = (float)(rand() % 10);
	}
	// Multiply the two matrices on the host
	start = clock();
	for (int i = 0; i < ITERS; i++) {
		addVectorCPU(a, b, ccpu, SIZE);
	}

	end = clock();
	tcpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / ITERS;
	// Display the results
	std::cout << "Host Computation took " << tcpu << " ms:" << std::endl;
	displayAddResults(a, b, ccpu);

	// Multiply the two matrices on the device
	// Perform one warm-up pass and validate
	bool success = addVectorGPU(a, b, cgpu, SIZE);
	if (!success) {
		std::cout << "\n * Device error! * \n" << std::endl;
		return 1;
	}
	// And now time it
	start = clock();
	for (int i = 0; i < ITERS; i++) {
		addVectorGPU(a, b, cgpu, SIZE);
	}
	end = clock();
	tgpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / ITERS;
	// Display the results
	std::cout << "Device Computation took " << tgpu << " ms:" << std::endl;
	displayAddResults(M, N, Pgpu);
	// Compare the results for correctness
	float sum = 0, delta = 0;
	for (int i = 0; i < SIZE*SIZE; i++) {
		delta += (Pcpu[i] - Pgpu[i]) * (Pcpu[i] - Pgpu[i]);
		sum += (Pcpu[i] * Pgpu[i]);
	}
	float L2norm = sqrt(delta / sum);
	std::cout << "Relative error: " << L2norm << "\n" << 
	((L2norm < 1e-6) ? "Passed" : "Failed") << std::endl;
	// Release the matrices
	delete[] a; delete[] b; delete[] ccpu; delete[] cgpu;
	// Success
	return 0;

}
*/
