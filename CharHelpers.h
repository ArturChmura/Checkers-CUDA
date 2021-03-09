#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

__host__ __device__  static char toLower(char c)
{
	if (c >= 'a')
	{
		return c;
	}
	return c - 'A' + 'a';
}

__host__ __device__  static char toUpper(char c)
{
	if (c >= 'a')
	{
		return c - 'a' + 'A';
	}
	return c;
	
}

/// <summary>
/// Get direction for player to move 
/// </summary>
__host__ __device__ static short GetDirection(char player)
{
	if (toLower(player) == 'w')
	{
		return 1;
	}
	return -1;
}

__host__ __device__  static char Opponent(char player)
{
	if (toLower(player) == 'w')
	{
		return 'b';
	}
	return 'w';
}

/// <summary>
/// Gets score for given result
/// </summary>
/// <param name="result">Game result</param>
/// <param name="player">Player to receive points</param>
/// <returns>Score</returns>
__host__ __device__ static int GetResultPoints(char result, char player)
{
	if (player == 'w' && result == GameResultWhitesWon) return  3;
	if (player == 'b' && result == GameResultBlacksWon) return  3;
	if (result == GameResultDraw) return 1;
	return 0;
}