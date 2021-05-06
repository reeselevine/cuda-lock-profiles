#include <stdio.h>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>

const int minWorkgroups = 2;
const int maxWorkgroups = 2;
const int numIterations = 10;
const int expectedCount = 20480;

// general
int* var;
int* d_var;

// spin lock
int* flag;
int* d_flag;

// petersons
int* level;
int* d_level;
int* victim;
int* d_victim;

__global__
void petersons(volatile int* _level, volatile int* _victim, int* _var, int numIterations) {
	if (threadIdx.x == 0) {
		for (int i = 0; i < numIterations; i++) {
			for (int j = 0; j < gridDim.x - 1; j++) {
				_level[blockIdx.x] = j;
				_victim[j] = blockIdx.x;
				for (int k = 0; k < gridDim.x; k++) {
					while (k != blockIdx.x && _level[k] >= j && _victim[j] == blockIdx.x);
				}
			}
			__threadfence();
			*_var = *_var + 1;
			__threadfence();
			_level[blockIdx.x] = -1;
		}
	}
}

__global__
void spinLock(volatile int* _flag, int* _var, int numIterations) {
	if (threadIdx.x == 0) {
		for (int i = 0; i < numIterations; i++) {
			while(atomicCAS((int*) _flag, 0, 1) == 1);
			__threadfence();
			*_var = *_var + 1;
			__threadfence();
			*_flag = 0;
		}
	}
}

void initializeBuffers(std::string testName) {
	var = (int*)malloc(1*sizeof(int));
	cudaMalloc(&d_var, 1*sizeof(int));
	if (testName == "spin-lock") {
		flag = (int*)malloc(1*sizeof(int));
		cudaMalloc(&d_flag, 1*sizeof(int));
	} else if (testName == "petersons") {
		level = (int*)malloc(maxWorkgroups*sizeof(int));
		cudaMalloc(&d_level, maxWorkgroups*sizeof(int));
		victim = (int*)malloc(maxWorkgroups*sizeof(int));
		cudaMalloc(&d_victim, maxWorkgroups*sizeof(int));
	}
}

void prepareBuffers(std::string testName) {
	if (testName == "spin-lock") {
		*flag = 0;
		cudaMemcpy(d_flag, flag, 1*sizeof(int), cudaMemcpyHostToDevice);
	} else if (testName == "petersons") {
		for (int i = 0; i < maxWorkgroups; i++) {
			level[i] = 0;
			victim[i] = 0;
		}
		cudaMemcpy(d_level, level, maxWorkgroups*sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(d_victim, victim, maxWorkgroups*sizeof(int), cudaMemcpyHostToDevice);
	}
}

void freeBuffers(std::string testName) {
	cudaFree(d_var);
	free(var);
	if (testName == "spin-lock") {
		cudaFree(d_flag);
		free(flag);
	} else if (testName == "petersons") {
		cudaFree(d_level);
		cudaFree(d_victim);
		free(level);
		free(victim);
	}
}

void runTest(std::string testName, int iterationsPerTest, int numWorkgroups) {
	if (testName == "spin-lock") {
		std::cout << "iterations per test: " << iterationsPerTest << "\n";
		spinLock<<<numWorkgroups, 1>>>(d_flag, d_var, iterationsPerTest);
	} else if (testName == "petersons") {
		petersons<<<numWorkgroups, 1>>>(d_level, d_victim, d_var, iterationsPerTest);
	}
}


int main(int argc, char* argv[]) {
	if (argc != 2) {
		std::cout << "Test name must be specified\n";
	}
	std::string testName(argv[1]);
	srand (time(NULL));

	std::cout << "Running Test" << testName << "\n";
	initializeBuffers(testName);
	double sum = 0;
	std::chrono::time_point<std::chrono::system_clock> start, end;
	for (int numWorkgroups = minWorkgroups; numWorkgroups <= maxWorkgroups; numWorkgroups*=2) {
		std::cout << "\nTest workgroups " << numWorkgroups << "\n";
		int iterationsPerTest = expectedCount/numWorkgroups;
		for (int i = 0; i < numIterations + 1; i++) {
			std::cout << "\ntest iteration " << i << "\n";
			*var = 0;
			cudaMemcpy(d_var, var, 1*sizeof(int), cudaMemcpyHostToDevice);
			prepareBuffers(testName);
			start = std::chrono::system_clock::now();
		        runTest(testName, iterationsPerTest, numWorkgroups);
			end = std::chrono::system_clock::now();
			cudaMemcpy(var, d_var, 1*sizeof(int), cudaMemcpyDeviceToHost);
			std::chrono::duration<double> result = end - start;
			if (i > 0) sum += result.count();
			std::cout << "iteration time: " << result.count() << "s\n";
			std::cout << "expected: " << expectedCount << ", actual: " << *var << "\n";
			if (expectedCount != *var) {
				std::cout << "Expected not equal to actual!\n";
			}
		}
		std::cout << "Average test iteration time: " << sum / numIterations << "s\n";
		sum = 0;
	}
	freeBuffers(testName);
	return 0;
}
