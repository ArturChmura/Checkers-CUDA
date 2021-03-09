#pragma once

#include <SFML/Graphics.hpp>
using namespace sf;
static Vector2f ShortenVector(Vector2f vector, float shortenDistance)
{
	float length = sqrt(vector.x * vector.x + vector.y * vector.y);
	vector *= (length - shortenDistance) / length;
	return vector;
}