#pragma once
#include "Node.h"
#include "RandomGame.h"

/// <summary>
/// Sequential rollout on CPU, fires one thread to rollout single state
/// </summary>
struct SequentialCpuNode : public Node
{
	SequentialCpuNode(Node* parentPtr, Move move) :Node(parentPtr, move)
	{
	}

	SequentialCpuNode(Board* boardPtr, char player) : Node(boardPtr, player)
	{
	}
	void Rollout() override
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
		return new SequentialCpuNode(this, move);
	}
};