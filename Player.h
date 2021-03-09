#pragma once
using namespace std;

struct Board;
struct Move;

//Base class for all players

class Player
{
public:
	char color;
	virtual Move GetMove(const Board* boardPtr, double timeLimitSec) = 0;

	
private:

};

