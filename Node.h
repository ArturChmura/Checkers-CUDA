#pragma once
#include <math.h>
#include <limits>
#include <iostream>
#include <random>
#include "SimpleStructs.h"
#include "Board.h"
#include <functional>
#include "NextFullMove.h"
#include "Results.h"
#include <vector>
#include <algorithm>

using namespace std;

static float c = 2; // Exploration parameter

struct Node
{
	Board* boardPtr = nullptr; // copy of board
	Move move; // last move before this state
	Node* parentPtr = nullptr; // parent Node
	Node** children = nullptr; 
	short childrenCount = -1;
	int visitedCount = 0; 
	int score = 0;
	char playerToMoveNext; 
	char player; // the player for whom the move is calculated
	bool isFullyEvaluated = false;  // indicates if node is fully evaluated (cannot be improved)

	Node(Node* parentPtr, Move move) :move(move), parentPtr(parentPtr)
	{
		this->boardPtr = new Board(parentPtr->boardPtr);
		playerToMoveNext = Opponent(parentPtr->playerToMoveNext);
		this->boardPtr->MakeMove(move);
		player = parentPtr->player;
	}

	Node(Board* boardPtr, char player) : parentPtr(nullptr), boardPtr(boardPtr), playerToMoveNext(player), player(player)
	{
	}

	/// <summary>
	/// Gets best move for given state
	/// </summary>
	Move GetBestMove() const
	{
		if (childrenCount <= 0)
		{
			auto moves = CalculatePossibleMoves(boardPtr, playerToMoveNext);
			return moves[0];
		}
		float maxWr = children[0]->WinRatio();
		Node* maxChild = children[0];
		for (short i = 1; i < childrenCount; i++)
		{
			float score = children[i]->WinRatio();
			if (score > maxWr)
			{
				maxWr = score;
				maxChild = children[i];
			}
		}

		return maxChild->move;

	}

	/// <summary>
	/// Evaluation function. If its visiting node first time, it rollouts game, else it picks best child for evaluation.
	/// </summary>
	virtual void Evaluate()
	{
		if (childrenCount == 0)
		{
			auto gameResult = GetFullResult(boardPtr);
			if (gameResult != GameResultNoResult)
			{
				PropagateBack(1, GetPoints(gameResult));
				return;
			}
		}
		if (childrenCount < 0)
		{
			if (visitedCount == 0 && parentPtr)
			{
				Rollout();
			}
			else
			{
				CreateChildren();
				if (childrenCount <= 1 && !parentPtr) // only one move from begging
				{
					isFullyEvaluated = true;
					return;
				}
				if (childrenCount > 0)
				{
					children[0]->Rollout();
				}
				else
				{
					auto gameResult = GetFullResult(boardPtr);
					PropagateBack(1, GetPoints(gameResult));
					return;
				}
			}
		}
		else
		{
			Node* maxChild = ChildWithMaxUCB();
			maxChild->Evaluate();
		}
	}

	~Node()
	{
		if (parentPtr)
		{
			delete boardPtr;
		}

		for (short i = 0; i < childrenCount; i++)
		{
			delete children[i];
		}
		delete[] children;
	}

	float UCB() const
	{
		if (visitedCount == 0)
		{
			return numeric_limits<float>::max();
		}
		return float(score) / visitedCount + c * sqrtf(logf(float(parentPtr->visitedCount) / visitedCount));
	}

	virtual void Rollout() = 0;

	virtual Node* GetChild(Node* parent, Move move) = 0;

	/// <summary>
	/// Gets vector of children that are best for rollout
	/// </summary>
	/// <param name="outNodes">Vector of nodes to which the children will be added</param>
	/// <param name="maxSize">Size of vector</param>
	void GetChildrenToRollout(vector<Node*>* outNodes, int maxSize)
	{
		if (outNodes->size() < maxSize)
		{
			if (visitedCount > 0)
			{
				if (childrenCount <= 0)
				{
					CreateChildren();
				}
				if (childrenCount <= 0)
				{
					outNodes->push_back(this);
					return;
				}
				std::sort(children, children + childrenCount, [&](Node* a, Node* b) {return a->UCB() > b->UCB(); });
				for (int i = 0; i < childrenCount && outNodes->size() < maxSize; i++)
				{
					children[i]->GetChildrenToRollout(outNodes, maxSize);
				}
			}
			else
			{
				outNodes->push_back(this);
			}

		}
	}

	
	float WinRatio() const
	{
		if (visitedCount == 0)
		{
			return 0;
		}
		return float(score) / visitedCount;
	}

	void PropagateBack(int trys, int points)
	{
		this->visitedCount += trys;
		this->score += points;
		if (parentPtr)
		{
			parentPtr->PropagateBack(trys, points);
		}
	}

	/// <summary>
	/// Creates nodes children, by calculating all possible moves
	/// </summary>
	void CreateChildren()
	{
		auto moves = CalculatePossibleMoves(boardPtr, playerToMoveNext);
		childrenCount = (short)moves.size();
		children = new Node * [childrenCount];

		for (short i = 0; i < childrenCount; ++i)
		{
			children[i] = GetChild(this, moves[i]);
		}
	}

	/// <summary>
	/// Gets childern witch maximum UCB
	/// </summary>
	Node* ChildWithMaxUCB() const
	{
		Node* maxChild = children[0];
		float maxUCB = maxChild->UCB();
		for (short i = 0; i < childrenCount; i++)
		{
			float ucb = children[i]->UCB();
			if (ucb > maxUCB)
			{
				maxUCB = ucb;
				maxChild = children[i];
			}
		}

		return maxChild;
	}

	/// <summary>
	/// Returns points by given result
	/// </summary>
	int GetPoints(char result)  const
	{
		return GetResultPoints(result, player);
	}




};

