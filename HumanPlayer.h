#pragma once

#include "Player.h"
#include <SFML/Graphics.hpp>
#include "SimpleStructs.h"
#include "CheckersDrawer.h"

using namespace std;
using namespace sf;


// Human player implemented in SFML

class HumanPlayer : public Player
{
public:
	HumanPlayer(short boardSize, RenderWindow* window, CheckersDrawer* drawer) : boardSize(boardSize), window(window), drawer(drawer)
	{
	}
	Move GetMove(const Board* boardPtr, double timeLimitSec) override
	{
		Move move;
		bool isMoveDone = false;
		while (!isMoveDone)
		{
			// check all the window's events that were triggered since the last iteration of the loop
			sf::Event event;
			while (window->pollEvent(event))
			{
				// "close requested" event: we close the window
				if (event.type == sf::Event::Closed)
					window->close();
				else if (event.type == Event::MouseButtonPressed)
				{
					Vector2i mousePosition =  Mouse::getPosition(*window);
					Position cellPosition = GetPositionOnBoard(mousePosition);
					move.AddNextPosition(cellPosition);
					MoveType moveType = boardPtr->GetMoveType(move,color);
					if(moveType == MoveType::Bad)
					{
						move = Move();
					}
					drawer->DrawBoard(boardPtr);
					drawer->HighliteMove(move);
					drawer->Display();
					if (moveType == MoveType::Good)
					{
						isMoveDone = true;
						break;
					}
				}
				else if (event.type == Event::Resized)
				{
					window->setView(sf::View(sf::FloatRect(0, 0, (float)event.size.width, (float)event.size.height)));
					drawer->DrawBoard(boardPtr);
					drawer->HighliteMove(move);
					drawer->Display();
				}

			}
		}
		
		return move;
	}
private:
	short boardSize;
	RenderWindow* window;
	CheckersDrawer* drawer;

	Vector2f CellSize() {
		auto windowSize = window->getSize();
		return { (float)windowSize.x / boardSize, (float)windowSize.y / boardSize };
	}

	Position GetPositionOnBoard(Vector2i mousePosition)
	{
		auto cellSize = CellSize();
		int row = (int)(boardSize - mousePosition.y / cellSize.y);
		int column = (int)(mousePosition.x / cellSize.x);
		return Position(row, column);
	}




};

