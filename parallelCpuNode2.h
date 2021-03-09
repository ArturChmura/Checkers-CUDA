#pragma once
#include "Node.h"
#pragma warning(push, 0)
#include <ppl.h>
#pragma warning(pop)
#include "RandomGame.h"
#include <algorithm>    // std::sort


/// <summary>
/// Parallel rollout on CPU, gets best nodes for rollout and then fires one thread for each
/// </summary>
struct ParallelCpuNode2 : public Node
{
	ParallelCpuNode2(Node* parentPtr, Move move) :Node(parentPtr, move)
	{
	}

	ParallelCpuNode2(Board* boardPtr, char player) : Node(boardPtr, player)
	{
	}

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

		const size_t processor_count = std::thread::hardware_concurrency();
		GetChildrenToRollout(&childrenToEv, processor_count);
		
		Concurrency::parallel_for(size_t(0),  childrenToEv.size(), [&](int i) {
			childrenToEv[i]->Rollout();
			});
	}

	void Rollout()
	{
		Board boardCpy = Board(boardPtr);
		char result;

		std::random_device rd;
		std::mt19937 gen = std::mt19937(rd());

		PlayRandomGame(&boardCpy, playerToMoveNext, &result, [&](int minIn, int maxEx) {
			std::uniform_int_distribution<> distr(0, maxEx - 1);
			return distr(gen);
			});

		short points = GetPoints(result);
		PropagateBack(1, points);
	}

	Node* GetChild(Node* parent, Move move) override
	{
		return new ParallelCpuNode2(this, move);
	}
};
