#pragma once
#include <stdint.h>
#include <stdlib.h>
#include <ctype.h>
#include "SimpleStructs.h"
#include "enums.h"
#include "CharHelpers.h"
using namespace std;
#include "CUDA_def.h"

__device__ __host__ static short sgn(short val) {
	return (0 < val) - (val < 0);
}

struct Board;

static void CalculateCapturingMoves(const Board* boardPtr, Position pawnPossition, vector<Move>* forcedMoves);
static void CalculateNonCapturingMoves(const Board* boardPtr, Position pawnPossition, vector<Move>* moves);
bool Contains(vector<Move>* moves, Move move);
static vector<Move> CalculateCapturingMoves(const Board* boardPtr, char player);
bool AnyBeginWith(vector<Move>* moves, Move move);

/// <summary>
/// Checkers board
/// </summary>
struct Board
{
	const short size;
	short whitesCount;
	short blacksCount;

	Board(short size) : size(size)
	{
		board = new char[(size_t)size * (size_t)size / 2];
		if (board == NULL) throw "Malloc error";
		for (short i = 0; i < size; i++)
		{
			for (short j = 0; j < size / 2; j++)
			{
				board[i * size / 2 + j] = ' ';
			}
		}
		for (short i = 0; i < size / 2 - 1; i++)
		{
			for (short j = 0; j < size / 2; j++)
			{
				board[i * size / 2 + j] = 'w';
				board[size * size / 2 - (i * size / 2 + j) - 1] = 'b';
			}
		}
		blacksCount = whitesCount = (size / 2 - 1) * (size / 2);
	}

	Board(const Board* boardPtr) : size(boardPtr->size)
	{
		this->board = new char[(size_t)size * (size_t)size / 2];
		if (this->board == NULL) throw "Malloc error";
		memcpy(this->board, boardPtr->board, size * size / 2 * sizeof(char));
		whitesCount = boardPtr->whitesCount;
		blacksCount = boardPtr->blacksCount;
	}

	bool device = false;
	short dataShift = 1;
	__device__  Board(char* boardPtr, short size) : size(size), board(boardPtr)
	{
		device = true;
		dataShift = BANKSCOUNT;
	}

	__device__ __host__  char GetPawn(short row, short column) const
	{
		if (IsUnusedField(row, column)) return ' ';
		return board[GetPawnPositionInArray(row, column)];
	}

	__device__ __host__ char GetPawn(Position position)  const
	{
		return  GetPawn(position.row, position.column);
	}

	__device__ __host__  void SetPawn(short row, short column, char pawn)
	{
		short pawnPos = GetPawnPositionInArray(row, column);
		char takenPawn = board[pawnPos];
		if (toLower(takenPawn) == 'w')
		{
			whitesCount--;
		}
		if (toLower(takenPawn) == 'b')
		{
			blacksCount--;
		}

		if (toLower(pawn) == 'w')
		{
			whitesCount++;
		}
		if (toLower(pawn) == 'b')
		{
			blacksCount++;
		}

		board[pawnPos] = pawn;
	}

	__device__ __host__ void SetPawn(Position position, char pawn) 
	{
		SetPawn(position.row, position.column, pawn);
	}

	__device__ __host__ bool IsEmpty(short row, short column)  const
	{
		return IsInBoud(row, column) && GetPawn(row, column) == ' ';
	}

	__device__ __host__ bool IsEmpty(Position position)  const 
	{
		return IsEmpty(position.row, position.column);
	}

	__device__ __host__ bool IsOpponent(short row, short column, char player) const
	{
		if (!IsInBoud(row, column)) return false;
		if (toLower(player) == 'w')
		{
			return toLower(GetPawn(row, column)) == 'b';
		}
		else if (toLower(player) == 'b')
		{
			return toLower(GetPawn(row, column)) == 'w';
		}

		return false;
	}

	__device__ __host__ bool IsOpponent(Position position, char player) const
	{
		return IsOpponent(position.row, position.column, player);
	}

	__host__ __device__ void MakeMove(Move move)
	{
		int moveSize = (int)move.positions.size();
		char pawn = GetPawn(move.positions[0]);
		if (pawn != toLower(pawn))
		{
			queenMovesWithoutCapturing++;
		}
		else
		{
			queenMovesWithoutCapturing = 0;
		}
		SetPawn(move.positions[0], ' ');
		for (short i = 0; i < moveSize - 1; i++)
		{
			Position start = move.positions[i];
			Position end = move.positions[i + 1];
			if (abs(start.row - end.row) >= 2)
			{
				Position taken = GetTakenPosition(start, end);
				if (!IsEmpty(taken)) //capturing
				{
					SetPawn(taken, ' ');
					queenMovesWithoutCapturing = 0;
				}
			}

			if ((pawn == 'w' && end.row == size - 1) || (pawn == 'b' && end.row == 0))
			{
				pawn = toUpper(pawn);
			}

		}
		SetPawn(move.positions[moveSize - 1], pawn);
	}

	__host__ __device__ void MakeMove(SimpleMove move)
	{
		Position start = move.start;
		Position end = move.end;
		char pawn = GetPawn(start);
		if (pawn != toLower(pawn))
		{
			queenMovesWithoutCapturing++;
		}
		else
		{
			queenMovesWithoutCapturing = 0;
		}
		SetPawn(start, ' ');


		if (IsMoveCapturing(start, end))
		{
			Position taken = GetTakenPosition(start, end);
			SetPawn(taken, ' ');
			queenMovesWithoutCapturing = 0;

		}

		if ((pawn == 'w' && end.row == size - 1) || (pawn == 'b' && end.row == 0))
		{
			pawn = toUpper(pawn);
		}

		SetPawn(end, pawn);
	}

	__device__ __host__ bool IsInBoud(short row, short column)  const
	{
		return row >= 0 && row < size&& column >= 0 && column < size;
	}

	__device__ __host__ bool IsInBoud(Position position) const
	{
		return IsInBoud(position.row, position.column);
	}

	__device__ __host__  bool IsDraw() const
	{
		return queenMovesWithoutCapturing >= 30;
	}

	__device__ __host__ bool IsMoveCapturing(Position start, Position end) const
	{
		if (abs(start.row - end.row) < 2) return false;
		Position takenPosition = GetTakenPosition(start, end);
		char takenPawn = GetPawn(takenPosition);
		return takenPawn != ' ';
	}

	__device__ __host__ MoveType GetMoveType(Move move,char player) const
	{
		int moveSize = (int)move.positions.size();
		if (moveSize == 0) return MoveType::Unfinished;

		char pawn = GetPawn(move.positions[0]);
		if (toLower(pawn) != player) return MoveType::Bad;
		if (moveSize == 1) return MoveType::Unfinished;
		vector<Move> possibleCapturingMoves = CalculateCapturingMoves(this, player);
		if (possibleCapturingMoves.size() > 0)
		{
			if (Contains(&possibleCapturingMoves, move)) return MoveType::Good;
			if (AnyBeginWith(&possibleCapturingMoves, move)) return MoveType::Unfinished;
		}
		else
		{
			vector<Move> possibleNonCapturingMoves;
			CalculateNonCapturingMoves(this, move.positions[0], &possibleNonCapturingMoves);
			if (Contains(&possibleNonCapturingMoves, move)) return MoveType::Good;
			if (AnyBeginWith(&possibleNonCapturingMoves, move)) return MoveType::Unfinished;
		}

		return MoveType::Bad;

	}

	__device__ __host__ bool IsPawn(char pawn) const
	{
		return toLower(pawn) == 'w' || toLower(pawn) == 'b';
	}

	__device__ __host__ ~Board()
	{
		if (!device)
		{
			free(board);
		}
	}


	char* board;

private:
	int queenMovesWithoutCapturing = 0;


	__device__ __host__ Position GetTakenPosition(Position start, Position end) const
	{
		short rowd = end.row - start.row;
		short cold = end.column - start.column;
		rowd -= sgn(rowd);
		cold -= sgn(cold);
		return start + Position(rowd, cold);
	}

	__device__ __host__ short GetPawnPositionInArray(short row, short column) const
	{
		return (row * size / 2 + column / 2)*dataShift;
	}
	__device__ __host__ bool IsUnusedField(short row, short column) const
	{
		return (((row + column) & 1) != 0);
	}


};
