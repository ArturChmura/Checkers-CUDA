#pragma once
struct Board;
struct Move;

/// <summary>
/// Abstract class to display game
/// </summary>
class CheckersDrawer
{
public:
	virtual void DrawBoard(const Board* board) = 0;
	virtual void DrawMove(Move move) = 0;
	virtual void Display() = 0;
	virtual void HighliteMove(Move move) = 0;

private:

};

