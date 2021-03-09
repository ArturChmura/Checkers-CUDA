#pragma once
#include "Player.h"
#include "sequentialCpuNode.h"
#include "parallelCpuNode.h"
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <ctime>
#include <ctype.h>
#include "SimpleStructs.h"
#include "Board.h"
#include "Node.h"
#include "parallelGpuNode.cuh"
#include "parallelGpuNode2.cuh"
#include "parallelCpuNode2.h"

// All computer players 

static Move CalculateBestMove(char player, double timeLimitSec, Node* node);

class ComputerPlayerSeq : public Player
{
public:
	Move GetMove(const Board* boardPtr, double timeLimitSec) override
	{
		Board boardCpy = Board(boardPtr);
		SequentialCpuNode node = SequentialCpuNode(&boardCpy, color);
		Move move = CalculateBestMove(color, timeLimitSec, &node);
		return move;
	}
};

class ComputerPlayerPar : public Player
{
public:
	Move GetMove(const Board* boardPtr, double timeLimitSec) override
	{
		Board boardCpy = Board(boardPtr);
		ParallelCpuNode node = ParallelCpuNode(&boardCpy, color);
		Move move = CalculateBestMove( color, timeLimitSec, &node);
		return move;
	}
};

class ComputerPlayerPar2 : public Player
{
public:
	Move GetMove(const Board* boardPtr, double timeLimitSec) override
	{
		Board boardCpy = Board(boardPtr);
		ParallelCpuNode2 node = ParallelCpuNode2(&boardCpy, color);
		Move move = CalculateBestMove(color, timeLimitSec, &node);
		return move;
	}
};

class ComputerPlayerGpu : public Player
{
public:
	Move GetMove(const Board* boardPtr, double timeLimitSec) override
	{
		Board boardCpy = Board(boardPtr);
		ParallelGpuNode node = ParallelGpuNode(&boardCpy, color);
		Move move = CalculateBestMove( color, timeLimitSec, &node );
		return move;
	}
};

class ComputerPlayerGpu2 : public Player
{
public:
	Move GetMove(const Board* boardPtr, double timeLimitSec) override
	{
		Board boardCpy = Board(boardPtr);
		ParallelGpuNode2 node = ParallelGpuNode2(&boardCpy, color);
		Move move = CalculateBestMove(color, timeLimitSec, &node);
		return move;
	}
};

static Move CalculateBestMove( char player, double timeLimitSec, Node* node)
{
	clock_t begin = clock();
	clock_t end = clock();
	int e = 0;
	while ((double(end) - double(begin)) / CLOCKS_PER_SEC < timeLimitSec && (node->isFullyEvaluated == false)/* || (double(end) - double(begin)) / CLOCKS_PER_SEC < 0.5)*/)
	{
		node->Evaluate();
		end = clock();
	}
	return node->GetBestMove();
}

