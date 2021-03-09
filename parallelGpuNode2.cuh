
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
#pragma warning(push, 0)
#include <ppl.h>
#pragma warning(pop)


void CallPlayRandomGameGPU2(dim3 numBlocks, dim3 threadsPerBlock, int n, int* results, char* boards, short boardSize, char* playersToMoveNext, char player);

/// <summary>
///  Parallel rollout on GPU. Gets best nodes for rollout and then fires one thread for each
/// </summary>
struct ParallelGpuNode2 : public Node
{
	ParallelGpuNode2(Node* parentPtr, Move move, int* results, char* boards, char* playersToMoveNext)
		:Node(parentPtr, move), results(results), boards(boards), playersToMoveNext(playersToMoveNext)
	{
	}

	ParallelGpuNode2(Board* boardPtr, char player) : Node(boardPtr, player)
	{
		gpuErrchk(cudaMalloc((void**)&results, THREADSCOUNT * sizeof(int)));
		gpuErrchk(cudaMalloc((void**)&boards, THREADSCOUNT * boardPtr->size * boardPtr->size / 2 * sizeof(char)));
		gpuErrchk(cudaMalloc((void**)&playersToMoveNext, THREADSCOUNT * sizeof(char)));
	}

	void Rollout()
	{
	}


	const size_t processor_count = THREADSCOUNT;
	int* results = nullptr;
	char* boards = nullptr; 
	char* playersToMoveNext = nullptr;
	int* resultsHost = new int[THREADSCOUNT];

	

	void Evaluate() override
	{

		if (childrenCount == 1)
		{
			isFullyEvaluated = true;
			return;
		}
		if (childrenCount == 0)
		{
			return;
		}

		vector<Node*> childrenToEv;
		GetChildrenToRollout(&childrenToEv, processor_count);;

		for (size_t i = 0; i < childrenToEv.size(); i++)
		{
			gpuErrchk(cudaMemcpy(boards + i * boardPtr->size * boardPtr->size / 2 * sizeof(char), childrenToEv[i]->boardPtr->board, boardPtr->size * boardPtr->size / 2 * sizeof(char), cudaMemcpyKind::cudaMemcpyHostToDevice));
		}
		for (size_t i = 0; i < childrenToEv.size(); i++)
		{
			gpuErrchk(cudaMemcpy(playersToMoveNext + i * sizeof(char), &childrenToEv[i]->playerToMoveNext, sizeof(char), cudaMemcpyKind::cudaMemcpyHostToDevice));
		}
		CallPlayRandomGameGPU2(numBlocks, threadsPerBlock, childrenToEv.size(), results, boards, boardPtr->size, playersToMoveNext, player);
		gpuErrchk(cudaMemcpy(resultsHost, results, childrenToEv.size() * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost));
		for (size_t i = 0; i < childrenToEv.size(); i++)
		{
			childrenToEv[i]->PropagateBack(1, resultsHost[i]);
		}
	}

	~ParallelGpuNode2()
	{
		if (!parentPtr)
		{
			gpuErrchk(cudaFree(results));
			gpuErrchk(cudaFree(boards));
			gpuErrchk(cudaFree(playersToMoveNext));
		}
	}

	Node* GetChild(Node* parent, Move move) override
	{
		return new ParallelGpuNode2(this, move,results,boards,playersToMoveNext);

	}
};

