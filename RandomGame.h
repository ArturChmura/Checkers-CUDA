#pragma once
#include "SimpleStructs.h"
#include "Board.h"
#include "NextSimpleMoves.h"


// Simulates random game from given state
template< typename Fn>
__device__ __host__ static void PlayRandomGame(Board* boardPtr, char playerToMoveNext, char* outResult, Fn rng)
{
	char result = GetResult(boardPtr);

	Deque<SimpleMove> moves;
	while (GameResultNoResult == result)
	{
		moves.clear();
		CalculatePossibleSimpleMoves(boardPtr, playerToMoveNext, &moves);
		int movesCount = moves.size();
		if (movesCount > 0)
		{

			short randomMoveInt = rng(0, movesCount);
			SimpleMove randomMove = moves[randomMoveInt];
			if (boardPtr->IsMoveCapturing(randomMove.start, randomMove.end))
			{
				boardPtr->MakeMove(randomMove);
				moves.clear();
				CalculateCapturingSimpleMoves(boardPtr, playerToMoveNext, &moves);
				int captMovesCount = moves.size();
				while (captMovesCount > 0)
				{
					short randomMoveInt = rng(0, movesCount);
					randomMove = moves[randomMoveInt];
					boardPtr->MakeMove(randomMove);
					moves.clear();
					CalculateCapturingSimpleMoves(boardPtr, playerToMoveNext, &moves);
					captMovesCount = moves.size();
				}
			}
			else
			{
				boardPtr->MakeMove(randomMove);
			}
		}
		else
		{
			result = Opponent(playerToMoveNext);
			break;
		}
		playerToMoveNext = Opponent(playerToMoveNext);
		result = GetResult(boardPtr);
	}
	*outResult = result;

}