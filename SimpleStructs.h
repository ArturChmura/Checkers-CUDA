#pragma once
#include <cstdint>
#include <stdlib.h>
#include <vector>

using namespace std;


struct Position
{
	short row = -1;
	short column = -1;

	__device__ __host__ Position(short row, short column) : row(row), column(column)
	{
	}

	__device__ __host__ Position operator+(const Position add) const
	{
		return Position(row + add.row, column + add.column);
	}

	__device__ __host__ bool operator==(const Position pos) const
	{
		return row == pos.row && column == pos.column;
	}
	__device__ __host__ bool operator!=(const Position pos) const
	{
		return !operator==(pos);
	}
	__device__ __host__ Position() {
	}
};

/// <summary>
/// Move with multiple capturing moves
/// </summary>
struct Move
{
	vector<Position> positions;

	Move(Position start, Position end)
	{
		positions.push_back(start);
		positions.push_back(end);
	}

	Move()
	{

	}

	void AddNextPosition(Position position)
	{
		positions.push_back(position);
	}
	Move operator+(Move move)
	{
		Move newMove = Move();
		for (size_t i = 0; i < positions.size(); i++)
		{
			newMove.positions.push_back(positions[i]);
		}
		for (size_t i = 1; i < move.positions.size(); i++)
		{
			newMove.positions.push_back(move.positions[i]);
		}
		return newMove;
	}
};

/// <summary>
/// Move withot multiple capturing moves
/// </summary>
struct SimpleMove
{
	Position start;
	Position end;

	__device__ __host__ SimpleMove(Position start, Position end)
	{
		this->start = start;
		this->end = end;
	}
};