#pragma once

#include <vector>
#include "SimpleStructs.h"
#include "Board.h"

/// <summary>
/// To compare Move objects
/// </summary>
struct compare
{
	Move key;
	compare(Move const& i) : key(i) { }

	bool operator()(Move const& cmpMove)
	{
		if (cmpMove.positions.size() != key.positions.size()) return false;
		bool same = true;
		for (size_t i = 0; i < key.positions.size(); i++)
		{
			if (key.positions[i] != cmpMove.positions[i])
			{
				same = false;
				break;
			}
		}
		return same;
	}
};

/// <summary>
/// Checks whether any of moves in vector is equal with given move
/// </summary>
static  bool Contains(vector<Move>* moves, Move move)
{
	return std::any_of(moves->begin(), moves->end(), compare(move));
}

/// <summary>
/// To check if one move begins with the other
/// </summary>
struct begins
{
	Move key;
	begins(Move const& i) : key(i) { }

	bool operator()(Move const& move)
	{
		if (key.positions.size() > move.positions.size())
		{
			return false;
		}

		bool begin = true;
		for (size_t j = 0; j < key.positions.size(); j++)
		{
			if (move.positions[j] != key.positions[j])
			{
				begin = false;
				break;
			}
		}

		return begin;

	}
};

/// <summary>
/// Checks whether any of moves in vector begins with given move
/// </summary>
static  bool AnyBeginWith(vector<Move>* moves, Move move)
{
	return std::any_of(moves->begin(), moves->end(), begins(move));
}

/// <summary>
/// Calculate all posible non capturing moves from given position and adds them to given vector
/// </summary>
/// <param name="boardPtr">Board pointer</param>
/// <param name="pawnPossition">Position of pawn to calculate moves for</param>
/// <param name="moves">Vector of moves to which the moves will be added</param>
static void CalculateNonCapturingMoves(const Board* boardPtr, Position pawnPossition, vector<Move>* moves)
{
	short row = pawnPossition.row;
	short col = pawnPossition.column;
	char pawn = boardPtr->GetPawn(pawnPossition);
	short dir = GetDirection(pawn);

	if (pawn == toLower(pawn)) // normal pawn
	{
		for (short i = -1; i < 2; i += 2)
		{
			if (boardPtr->IsEmpty(row + dir, col + i))
			{
				moves->push_back(Move(pawnPossition, Position(row + dir, col + i)));
			}
		}
	}
	else // queen
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
/// <returns>Vecor of all possible capturing moves from given position</returns>
static vector<Move> CalculateNextCapturingMoves(const Board* boardPtr, Position pawnPossition)
{
	short row = pawnPossition.row;
	short col = pawnPossition.column;
	char pawn = boardPtr->GetPawn(pawnPossition);
	short dir = GetDirection(pawn);
	vector<Move> forcedMoves;
	if (pawn == toLower(pawn)) // normal pawn
	{
		for (short i = -1; i < 2; i += 2)
		{
			for (short j = -1; j < 2; j += 2)
			{
				if (boardPtr->IsOpponent(row + i, col + j, pawn) && boardPtr->IsEmpty(row + 2 * i, col + 2 * j))
				{
					Position nextPosition = Position(row + 2 * i, col + 2 * j);
					Move movesInThisDir = { pawnPossition,  nextPosition };
					Board boardAfterMove(boardPtr);
					boardAfterMove.MakeMove(movesInThisDir);
					vector<Move> nextMoves = CalculateNextCapturingMoves(&boardAfterMove, nextPosition);
					if (nextMoves.size() > 0)
					{
						for (size_t i = 0; i < nextMoves.size(); i++)
						{
							forcedMoves.push_back(movesInThisDir + nextMoves[i]);
						}
					}
					else
					{
						forcedMoves.push_back(movesInThisDir);
					}

				}
			}
		}

	}
	else // queen
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
					Move movesInThisDir = { pawnPossition,  nextPosition };
					Board boardAfterMove(boardPtr);
					boardAfterMove.MakeMove(movesInThisDir);
					vector<Move> nextMoves = CalculateNextCapturingMoves(&boardAfterMove, nextPosition);
					if (nextMoves.size() > 0)
					{
						for (size_t i = 0; i < nextMoves.size(); i++)
						{
							forcedMoves.push_back(movesInThisDir + nextMoves[i]);
						}
					}
					else
					{
						forcedMoves.push_back(movesInThisDir);
					}
				}


			}
		}

	}

	return forcedMoves;
}


/// <summary>
/// Calculate all posible capturing moves from given position
/// </summary>
/// <param name="boardPtr">Board pointer</param>
/// <param name="pawnPossition">Position of pawn to calculate moves for</param>
/// <param name="forcedMoves">Vector of moves to which the moves will be added</param>
static void CalculateCapturingMoves(const Board* boardPtr, Position pawnPossition, vector<Move>* forcedMoves)
{
	auto moves = CalculateNextCapturingMoves(boardPtr, pawnPossition);
	for (size_t i = 0; i < moves.size(); i++)
	{
		forcedMoves->push_back(moves[i]);
	}
}

/// <summary>
/// Calculate all posible capturing moves for given player
/// </summary>
/// <param name="boardPtr">Board pointer</param>
/// <param name="player">Player to calculate moves for</param>
/// <returns>Vector of capturing moves</returns>
static vector<Move> CalculateCapturingMoves(const Board* boardPtr, char player)
{
	vector<Move> capturingMoves;
	for (short i = 0; i < boardPtr->size; i++)
	{
		for (short j = 0; j < boardPtr->size; j++)
		{
			if (toLower(boardPtr->GetPawn(i, j)) == player)
			{
				CalculateCapturingMoves(boardPtr, Position(i, j), &capturingMoves);
			}
		}
	}
	return capturingMoves;
}

/// <summary>
/// Calculate all posible non capturing moves for given player
/// </summary>
/// <param name="boardPtr">Board pointer</param>
/// <param name="player">Player to calculate moves for</param>
/// <returns>Vector of non capturing moves</returns>
static vector<Move> CalculateNonCapturingMoves(const Board* boardPtr, char player)
{
	vector<Move> nonCapturingMoves;
	for (short i = 0; i < boardPtr->size; i++)
	{
		for (short j = 0; j < boardPtr->size; j++)
		{
			if (toLower(boardPtr->GetPawn(i, j)) == player)
			{
				CalculateNonCapturingMoves(boardPtr, Position(i, j), &nonCapturingMoves);
			}
		}
	}
	return nonCapturingMoves;
}

/// <summary>
/// Calculate all posible moves for given player
/// </summary>
/// <param name="boardPtr">Board pointer</param>
/// <param name="player">Player to calculate moves for</param>
/// <returns>Vector of possible moves</returns>
static vector<Move> CalculatePossibleMoves(const Board* boardPtr, char player)
{
	auto capturingMoves = CalculateCapturingMoves(boardPtr, player);
	if (capturingMoves.size() > 0) return capturingMoves;
	auto nonCapturingMoves = CalculateNonCapturingMoves(boardPtr, player);
	return nonCapturingMoves;
}