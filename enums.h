#pragma once

using namespace std;

	

	static const char GameResultWhitesWon = 'w';
	static const char GameResultBlacksWon = 'b';
	static const char GameResultDraw = 'd';
	static const char GameResultNoResult = '!';

	enum class MoveType
	{
		Good = 'g',
		Bad = 'b',
		Unfinished = 'u',
	};