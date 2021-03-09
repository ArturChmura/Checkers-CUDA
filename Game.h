#pragma once
#include <stdint.h>
#include <stdlib.h>
#include "Board.h"
#include "enums.h"
#include "Player.h"
#include "CheckersDrawer.h"


using namespace std;

	class Game
	{
	public:

		double timeLimitSec;
		Player* player1;
		Player* player2;
		Board* board;
		Move lastMove;

		char result = GameResultNoResult;

		Game(Player* player1, Player* player2, Board* board,CheckersDrawer* drawer, double timeLimitSec) : player1(player1), player2(player2), board(board), timeLimitSec(timeLimitSec), drawer(drawer)
		{
			(*player1).color = 'w';
			(*player2).color = 'b';
		}
		void Start()
		{
			drawer->DrawBoard(board);
			drawer->Display();
			while (true)
			{
				PlayerTurn(player1);
				result = GetFullResult(board);
				if (result != GameResultNoResult)
				{
					break;
				}

				PlayerTurn(player2);
				result = GetFullResult(board);
				if (result != GameResultNoResult)
				{
					break;
				}
			}
		}

	private:
		CheckersDrawer* drawer;
		void MakeMove(Move move)
		{
			board->MakeMove(move);
			lastMove = move;
			drawer->DrawBoard(board);
			drawer->DrawMove(move);
			drawer->Display();
		}
		
		void PlayerTurn(Player* player)
		{
			Move move = player->GetMove(board, timeLimitSec);

			MoveType moveType = board->GetMoveType(move,player->color);
			if (moveType != MoveType::Good)
			{
				throw "badMove";
			}
			MakeMove(move);
		}
	};



