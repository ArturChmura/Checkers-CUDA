#pragma once
#include "Board.h"

/// <summary>
/// Return game result without draw by no possible moves
/// </summary>
/// <param name="boardPtr">Board pointer</param>
/// <returns>Game result</returns>
__device__  __host__  static char GetResult(const Board* boardPtr)
{
	if (boardPtr->IsDraw())
	{
		return GameResultDraw;
	}

	if (boardPtr->blacksCount == 0 && boardPtr->whitesCount > 0)
	{
		return GameResultWhitesWon;
	}
	if (boardPtr->blacksCount > 0 && boardPtr->whitesCount == 0)
	{
		return GameResultBlacksWon;
	}

	return GameResultNoResult;
}


/// <summary>
/// Return game result
/// </summary>
/// <param name="boardPtr">Board pointer</param>
/// <returns>Game result</returns>
__host__  static char GetFullResult(const Board* board)
{
	if (board->IsDraw())
	{
		return GameResultDraw;
	}

	if (board->blacksCount == 0 && board->whitesCount > 0)
	{
		return GameResultWhitesWon;
	}
	if (board->blacksCount > 0 && board->whitesCount == 0)
	{
		return GameResultBlacksWon;
	}

	auto moves = CalculatePossibleMoves(board, 'b');
	if (moves.size() == 0)
	{
		return GameResultWhitesWon;
	}

	moves = CalculatePossibleMoves(board, 'w');
	if (moves.size() == 0)
	{
		return GameResultBlacksWon;
	}

	return GameResultNoResult;
}
