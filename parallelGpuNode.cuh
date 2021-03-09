
#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <iostream>  
#include <math.h>
#include <thrust\device_ptr.h>
#include <thrust\sequence.h>
#include <device_launch_parameters.h>
#include <curand.h>
#include <curand_kernel.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include "Node.h"
#include "Deque.h"
#include "CudaErrorHelpers.h"

#define THREADSCOUNT 2048

int CallPlayRandomGameGPU(dim3 numBlocks, dim3 threadsPerBlock,  int n, int* results, char* boards, short boardSize, char playerToMoveNext, char player);

/// <summary>
///  Parallel rollout on GPU. Fires thousands of threads for single node
/// </summary>
struct ParallelGpuNode : public Node
{
	int* results = nullptr;

	char* boards = nullptr;

	ParallelGpuNode(Node* parentPtr, Move move, int* results, char* boards) :Node(parentPtr, move), results(results), boards(boards)
	{
	}

	ParallelGpuNode(Board* boardPtr, char player) : Node(boardPtr, player)
	{
		size_t resultsMallocSize = THREADSCOUNT * sizeof(int);
		gpuErrchk(cudaMalloc((void**)&results, resultsMallocSize));
		gpuErrchk(cudaMalloc((void**)&boards, boardPtr->size * boardPtr->size / 2 * sizeof(char)));
	}

		
	void Rollout()
	{
		gpuErrchk(cudaMemcpy(boards , boardPtr->board, boardPtr->size * boardPtr->size / 2 * sizeof(char), cudaMemcpyKind::cudaMemcpyHostToDevice));
		
		int pointSum = CallPlayRandomGameGPU (numBlocks, threadsPerBlock, THREADSCOUNT, results, boards, boardPtr->size, playerToMoveNext,player);
	
		PropagateBack(THREADSCOUNT, pointSum);

	}

	~ParallelGpuNode()
	{
		if (!parentPtr)
		{
			gpuErrchk(cudaFree(results));
			gpuErrchk(cudaFree(boards));
		}
	}


	Node* GetChild(Node* parent, Move move) override
	{
		return new ParallelGpuNode(this, move,results,boards);
	}
};

