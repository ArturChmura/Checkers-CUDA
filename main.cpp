#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "ComputerPlayer.h"
#include "Game.h"
#include "Drawer2D.h"
#include "HumanPlayer.h"

using namespace std;

void ChoosePlayer(Player** p1, int boardSize, RenderWindow* window, string name, CheckersDrawer* drawer)
{
	int choice = 0;
	cout << "Choose " << name << " player by typing number" << endl;
	cout << "1. Human Player" << endl;
	cout << "2. Sequential CPU Player" << endl;
	cout << "3. Parallel CPU Player" << endl;
	cout << "4. Parallel GPU Player [Nvidia GPUs users only]" << endl;
	cout << "5. Parallel CPU Player V2" << endl;
	cout << "6. Parallel GPU Player V2 [Nvidia GPUs users only]" << endl;


	while (choice <= 0 || choice >= 7)
	{
		while (!(cin >> choice) || (choice <= 0 || choice >= 7)) {
			cin.clear();
			cin.ignore(numeric_limits<streamsize>::max(), '\n');
			cout << "It's not that hard, just type corrent number" << endl;
		}
	}

	switch (choice)
	{
	case 1:
		(*p1) = new HumanPlayer(boardSize, window, drawer);
		break;
	case 2:
		(*p1) = new  ComputerPlayerSeq();
		break;
	case 3:
		(*p1) = new  ComputerPlayerPar();
		break;
	case 4:
		(*p1) = new  ComputerPlayerGpu();
		break;
	case 5:
		(*p1) = new  ComputerPlayerPar2();
		break;
	case 6:
		(*p1) = new  ComputerPlayerGpu2();
		break;
	default:
		break;
	}
}

void SetTimeLimit(double* timeLimit)
{
	double choice = 0;
	cout << "Write time limit for computer in seconds (higher = harder)" << endl;
	while (choice <= 0)
	{
		while (!(cin >> choice) || choice <= 0) {
			cin.clear();
			cin.ignore(numeric_limits<streamsize>::max(), '\n');
			cout << "It's not that hard, just type corrent number" << endl;
		}
	}

	*timeLimit = choice;
}

void SetBoardSize(int* timeLimit)
{
	int choice = 0;
	cout << "Write the board size (even  and >4)" << endl;
	while (choice <= 4 || choice % 2 == 1)
	{
		while (!(cin >> choice) || (choice <= 4 || choice % 2 == 1)) {
			cin.clear();
			cin.ignore(numeric_limits<streamsize>::max(), '\n');
			cout << "It's not that hard, just type corrent number" << endl;
		}
	}

	*timeLimit = choice;
}

short main()
{
	int windowSize = 800;
	RenderWindow window(sf::VideoMode(windowSize, windowSize), "Checkers!");
	window.setVisible(false);

	int boardSize;
	SetBoardSize(&boardSize);

	CheckersDrawer* drawer = new Drawer2D(boardSize, &window);

	Player* p1;
	Player* p2;

	ChoosePlayer(&p1, boardSize, &window, "white", drawer);
	ChoosePlayer(&p2, boardSize, &window, "black", drawer);

	double moveTimeLimit;
	SetTimeLimit(&moveTimeLimit);


	Board board = Board(boardSize);
	Game game = Game(p1, p2, &board, drawer, moveTimeLimit);

	window.setVisible(true);

	game.Start();

	char result = game.result;
	if (result == GameResultBlacksWon)
	{
		cout << "Blacks Won" << endl;
	}
	else if (result == GameResultWhitesWon)
	{
		cout << "Whites Won" << endl;
	}
	else if (result == GameResultDraw)
	{
		cout << "Draw" << endl;
	}

	while (window.isOpen())
	{
		// check all the window's events that were triggered since the last iteration of the loop
		sf::Event event;
		while (window.pollEvent(event))
		{
			// "close requested" event: we close the window
			if (event.type == sf::Event::Closed)
				window.close();
		}
		drawer->DrawBoard(&board);
		drawer->DrawMove(game.lastMove);
		drawer->Display();
	}
	delete p1, p2, drawer;
	return 0;
}