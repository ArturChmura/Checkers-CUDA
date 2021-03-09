#pragma once
#include "Node.h"
#pragma warning(push, 0)
#include <ppl.h>
#pragma warning(pop)
#include "RandomGame.h"


/// <summary>
/// Parallel rollout on CPU, fires all possible threads to rollout single state many times
/// </summary>
struct ParallelCpuNode : public Node
{
	ParallelCpuNode(Node* parentPtr, Move move) :Node(parentPtr,move)
	{
	}

	ParallelCpuNode(Board* boardPtr, char player) :  Node(boardPtr, player)
	{
	}

	void Rollout()
	{

		const auto processor_count = std::thread::hardware_concurrency();
		char* results = new char[processor_count];
		std::random_device rd; // obtain a random number from hardware
		std::mt19937 gen = std::mt19937(rd()); // seed the generator

		Board boardCpy = Board(boardPtr);
		Concurrency::parallel_for(size_t(0), size_t(processor_count), [&](int i) {
			PlayRandomGame(&boardCpy, playerToMoveNext, &results[i], [&](int minIn, int maxEx) {
				std::uniform_int_distribution<> distr(0, maxEx - 1);
				return distr(gen);
				});
			});
		int pointSum = 0;
		for (size_t i = 0; i < processor_count; i++)
		{
			pointSum += GetPoints(results[i]);
		}
		PropagateBack(processor_count, pointSum);
	}



	Node* GetChild(Node* parent, Move move) override
	{
		return new ParallelCpuNode(this, move);
	
	}
};
