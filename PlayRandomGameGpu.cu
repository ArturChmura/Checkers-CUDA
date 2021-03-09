
#pragma once
#pragma warning(push, 0)
#include <stdio.h>
#include <stdlib.h>
#include <iostream>  
#include <math.h>
#include <thrust\device_ptr.h>
#include <thrust\sequence.h>
#include <device_launch_parameters.h>
#include <curand.h>
#include <curand_kernel.h>
#pragma warning(pop)
#include "parallelGpuNode.cuh"
#include "Node.h"
#include "RandomGame.h"
#include "CUDA_def.h"


__global__ void PlayRandomGameGPU(int n, int* results, char* boards, short boardSize, char playerToMoveNext,char player)
{
	int i = blockIdx.x * blockDim.x * blockDim.y * blockDim.z + blockDim.x * blockDim.y * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x;
	if (i >= n) return;

	char* cBoard = boards ;

	//SHARED MEMORY
	extern  __shared__ char s[];
	size_t offset = (blockDim.x * blockDim.y * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x);
	char* sharedBoard = s + offset;
	for (size_t i = 0; i < boardSize * boardSize / 2; i++) // sets single board in one shared memory bank to minimalize confilcts
	{
		sharedBoard[i * BANKSCOUNT] = cBoard[i];
	}
	Board board = Board(sharedBoard, boardSize);
	//~~SHARED MEMORY


	curandState_t state;
	curand_init(0, 	i,1,	&state);
	char result;

	PlayRandomGame(&board, playerToMoveNext, &result, [&]__device__(int minIn, int maxEx) {
		return curand(&state) % maxEx + minIn;
		});

	results[i] = GetResultPoints(result,player);

}

int CallPlayRandomGameGPU(dim3 numBlocks, dim3 threadsPerBlock,  int n, int* results, char* boards, short boardSize, char playerToMoveNext, char player)
{
	size_t sharedMemorySize = threadsPerBlock.x* threadsPerBlock.y* threadsPerBlock.z * boardSize * boardSize / 2 * sizeof(char);
	PlayRandomGameGPU << <numBlocks, threadsPerBlock , sharedMemorySize >> > (n, results, boards, boardSize, playerToMoveNext,player);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	thrust::device_ptr<int> cptr = thrust::device_pointer_cast(results);
	int pointSum = thrust::reduce(cptr, cptr + THREADSCOUNT);
	return pointSum;
}

