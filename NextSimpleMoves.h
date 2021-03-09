#pragma once
#include "SimpleStructs.h"
#include "Board.h"
#include "Deque.h"


/// <summary>
/// Calculate all posible non capturing moves from given position and adds them to given vector
/// </summary>
/// <param name="boardPtr">Board pointer</param>
/// <param name="pawnPossition">Position of pawn to calculate moves for</param>
/// <param name="moves">Vector of moves to which the moves will be added</param>
__device__ __host__ static void CalculateNonCapturingSimpleMoves(const Board* boardPtr, Position pawnPossition, Deque<SimpleMove>* moves)
{
	short row = pawnPossition.row;
	short col = pawnPossition.column;
	char pawn = boardPtr->GetPawn(pawnPossition);
	short dir = GetDirection(pawn);

	if (pawn == toLower(pawn)) //normalny pionek
	{
		for (short i = -1; i < 2; i += 2)
		{
			if (boardPtr->IsEmpty(row + dir, col + i))
			{
				moves->push_back(SimpleMove(pawnPossition, Position(row + dir, col + i)));
			}
		}
	}
	else // królowa
	{
		for (short i = -1; i < 2; i += 2)
		{
			for (short j = -1; j < 2; j += 2)
			{
				Position pos = Position(row + i, col + j);
				while (boardPtr->IsEmpty(pos))
				{
					moves->push_back({ pawnPossition, pos });
					pos = pos + Position(i, j);
				}
			}
		}

	}
}

/// <summary>
/// Calculate all posible capturing moves from given position
/// </summary>
/// <param name="boardPtr">Board pointer</param>
/// <param name="pawnPossition">Position of pawn to calculate moves for</param>
/// <param name="forcedMoves">Vector of moves to which the moves will be added</param>
__device__ __host__   static void CalculateCapturingSimpleMoves(const Board* boardPtr, Position pawnPossition, Deque<SimpleMove>* forcedMoves)
{
	short row = pawnPossition.row;
	short col = pawnPossition.column;
	char pawn = boardPtr->GetPawn(pawnPossition);
	if (pawn == toLower(pawn)) //normalny pionek
	{
		for (short i = -1; i < 2; i += 2)
		{
			for (short j = -1; j < 2; j += 2)
			{
				if (boardPtr->IsOpponent(row + i, col + j, pawn) && boardPtr->IsEmpty(row + 2 * i, col + 2 * j))
				{
					Position nextPosition = Position(row + 2 * i, col + 2 * j);
					forcedMoves->push_back({ pawnPossition,  nextPosition });
				}
			}
		}

	}
	else // królowa
	{
		for (short i = -1; i < 2; i += 2)
		{
			for (short j = -1; j < 2; j += 2)
			{
				Position pos = Position(row + i, col + j);
				while (boardPtr->IsEmpty(pos))
				{
					pos = pos + Position(i, j);
				}

				if (boardPtr->IsOpponent(pos, pawn) && boardPtr->IsEmpty(pos + Position(i, j)))
				{
					Position nextPosition = pos + Position(i, j);
					forcedMoves->push_back({ pawnPossition,  nextPosition });
					break;
				}
			}
		}

	}

}

/// <summary>
/// Calculate all posible capturing moves for given player
/// </summary>
/// <param name="boardPtr">Board pointer</param>
/// <param name="player">Player to calculate moves for</param>
/// <param name="moves">Vector of moves to which the moves will be added</param>
__device__ __host__ static void CalculateCapturingSimpleMoves(const Board* boardPtr, char player, Deque<SimpleMove>* moves)
{
	for (short i = 0; i < boardPtr->size; i++)
	{
		for (short j = 0; j < boardPtr->size; j++)
		{
			if (toLower(boardPtr->GetPawn(i, j)) == player)
			{
				CalculateCapturingSimpleMoves(boardPtr, Position(i, j), moves);
			}
		}
	}
}


/// <summary>
/// Calculate all posible non capturing moves for given player
/// </summary>
/// <param name="boardPtr">Board pointer</param>
/// <param name="player">Player to calculate moves for</param>
/// <param name="moves">Vector of moves to which the moves will be added</param>
__device__ __host__ static void CalculateNonCapturingSimpleMoves(const Board* boardPtr, char player, Deque<SimpleMove>* moves)
{
	for (short i = 0; i < boardPtr->size; i++)
	{
		for (short j = 0; j < boardPtr->size; j++)
		{
			if (toLower(boardPtr->GetPawn(i, j)) == player)
			{
				CalculateNonCapturingSimpleMoves(boardPtr, Position(i, j), moves);
			}
		}
	}
}



/// <summary>
/// Calculate all posible moves for given player
/// </summary>
/// <param name="boardPtr">Board pointer</param>
/// <param name="player">Player to calculate moves for</param>
/// <param name="moves">Vector of moves to which the moves will be added</param>
__device__ __host__ static void CalculatePossibleSimpleMoves(const Board* boardPtr, char player, Deque<SimpleMove>* moves)
{
	CalculateCapturingSimpleMoves(boardPtr, player, moves);
	if (moves->size() > 0) return;
	CalculateNonCapturingSimpleMoves(boardPtr, player, moves);
	return;
}