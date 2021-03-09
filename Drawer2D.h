#pragma once
#include <SFML/Graphics.hpp>
#include "CheckersDrawer.h"
#include "Board.h"
#include "ThiccLine.h"
#include "SFMLHelpers.h"
#include <thread>       
#define M_PI 3.14159265358979323846

using namespace std;
using namespace sf;

/// <summary>
/// SFML implementation of CheckersDrawer
/// </summary>
class Drawer2D : public CheckersDrawer
{
public:
	short boardSize;
	Drawer2D(short boardSize, RenderWindow* window) : boardSize(boardSize), window(window)
	{
	}
	virtual void DrawBoard(const Board* board) override
	{
		if (window->isOpen())
		{
			window->clear();
			DrawBoardColors(board);
			DrawPawns(board);
		}
	}

	virtual void DrawMove(Move move) override
	{
		if (window->isOpen())
		{
			for (int i = 0; i < (int)move.positions.size() - 1; i++)
			{
				DrawArrow(move.positions[i], move.positions[i + 1]);
			}
		}
	}

	virtual void Display() override
	{
		if (window->isOpen())
		{
			window->display();
		}
		
	}

	virtual void HighliteMove(Move move) override
	{
		auto cellSize = CellSize();
		sf::RectangleShape rectangle(cellSize);
		rectangle.setFillColor(sf::Color(248, 252, 3,100));
		for (size_t i = 0; i < move.positions.size(); i++)
		{
			rectangle.setPosition(sf::Vector2f(move.positions[i].column * cellSize.x, (boardSize - move.positions[i].row - 1) * cellSize.y ));
			window->draw(rectangle);
		}
		
	}


	~Drawer2D()
	{
		window->close();
		delete window;
	}

private:
	RenderWindow* window;

	Vector2f CellSize() {
		auto windowSize = window->getSize();
		return { (float)windowSize.x / boardSize, (float)windowSize.y / boardSize };
	}

	Vector2f GetCellCenter(short row, short column)
	{
		auto cellSize = CellSize();
		return Vector2f((column + 0.5f) * cellSize.x, (boardSize - (row + 0.5f)) * cellSize.y);
	}
	void DrawArrow(Position p1, Position p2)
	{
		auto cellSize = CellSize();
		sf::Color color(217, 214, 65);
		Vector2f arrowStart = GetCellCenter(p1.row, p1.column);
		Vector2f arrowEnd = GetCellCenter(p2.row, p2.column);
		Vector2f vec = arrowEnd - arrowStart;
		sfLine line(arrowStart, arrowStart + ShortenVector(vec, cellSize.x / 4), cellSize.y / 8, color);
		window->draw(line);

		sf::ConvexShape triangle;
		triangle.setPointCount(3);
		triangle.setPoint(0, sf::Vector2f(0, 0));
		triangle.setPoint(1, sf::Vector2f(-cellSize.x / 2, -cellSize.y / 4));
		triangle.setPoint(2, sf::Vector2f(-cellSize.x / 2, cellSize.y / 4));

		triangle.setFillColor(color);
		float dx = arrowEnd.x - arrowStart.x;
		float dy = arrowEnd.y - arrowStart.y;
		float angle = atanf(dy / dx);
		if (dx < 0) angle += (float)M_PI;

		triangle.rotate(angle / (float)M_PI * 180);
		triangle.setPosition(arrowEnd);
		window->draw(triangle);
	}



	void DrawBoardColors(const Board* board)
	{
		short boardSize = board->size;
		if (window->isOpen())
		{
			for (short i = 0; i < boardSize; i++)
			{
				for (short j = 0; j < boardSize; j++)
				{
					sf::RectangleShape rectangle(CellSize());
					if (((i + j) & 1) == 0)
					{
						rectangle.setFillColor(sf::Color(207, 209, 208));
					}
					else
					{
						rectangle.setFillColor(sf::Color(58, 61, 60));
					}
					rectangle.setPosition(sf::Vector2f(j * CellSize().x, i * CellSize().y));
					window->draw(rectangle);
				}

			}
		}
	}

	void DrawPawns(const Board* board)
	{
		auto cellSize = CellSize();
		if (window->isOpen())
		{
			for (short i = 0; i < boardSize; i++)
			{
				for (short j = 0; j < boardSize; j++)
				{
					char pawn = board->GetPawn(i, j);
					if (pawn != ' ')
					{
						DrawPawnShape(pawn, cellSize,i,j);
						
					}
				}

			}
		}
	}



	void DrawPawnShape(char pawn, Vector2f cellSize, int row, int column)
	{
		if (pawn == toLower(pawn)) //normal pawn
		{
			CircleShape shape = GetPawn(cellSize);
			if (toLower(pawn) == 'w')
			{
				shape.setFillColor(sf::Color(255, 255, 255));
				shape.setOutlineColor(sf::Color(0, 0, 0));
			}
			else
			{
				shape.setFillColor(sf::Color(0, 0, 0));
				shape.setOutlineColor(sf::Color(255, 255, 255));
			}
			shape.setPosition(sf::Vector2f(column * cellSize.x + cellSize.x/2 - shape.getRadius() * shape.getScale().x, (boardSize - row - 1 ) * cellSize.y + cellSize.y / 2 - shape.getRadius() * shape.getScale().y) );
			shape.setOutlineThickness(5);
			window->draw(shape);
		}
		else
		{
			ConvexShape shape = GetQueen(cellSize);
			if (toLower(pawn) == 'w')
			{
				shape.setFillColor(sf::Color(255, 255, 255));
			}
			else
			{
				shape.setFillColor(sf::Color(0, 0, 0));
			}
			shape.setOutlineColor(sf::Color(218, 165, 32));
			shape.setPosition(sf::Vector2f(column* cellSize.x , (boardSize - row - 1)* cellSize.y ));
			shape.setOutlineThickness(5);
			window->draw(shape);
		}
		


	}

	CircleShape GetPawn(Vector2f cellSize)
	{
		CircleShape pawnShape = CircleShape(cellSize.x / 3);
		pawnShape.setScale(1, cellSize.y / cellSize.x);
		return pawnShape;
	}
	ConvexShape GetQueen(Vector2f cellSize)
	{
		sf::ConvexShape convex = ConvexShape(7);
		convex.setPoint(0, sf::Vector2f(cellSize.x / 8, cellSize.x / 8));
		convex.setPoint(1, sf::Vector2f(cellSize.x / 4, cellSize.x / 2));
		convex.setPoint(2, sf::Vector2f(cellSize.x / 2, cellSize.x / 8));
		convex.setPoint(3, sf::Vector2f(cellSize.x * 3 / 4, cellSize.x / 2));
		convex.setPoint(4, sf::Vector2f(cellSize.x - cellSize.x / 8, cellSize.x / 8));
		convex.setPoint(5, sf::Vector2f(cellSize.x - cellSize.x / 8, cellSize.x - cellSize.x / 8));
		convex.setPoint(6, Vector2f(cellSize.x / 8, cellSize.x - cellSize.x / 8));
		convex.setScale(1, cellSize.y / cellSize.x);
		return convex;
	}
};

