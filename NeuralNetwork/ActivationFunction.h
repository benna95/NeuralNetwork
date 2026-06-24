#pragma once
#include <math.h>
#include <string>

enum class TipoDiFunzione : int
{
	Linear				= 0,
	ReLU				= 1,
	Sigmoid				= 2,
	Tanh				= 3,
	Quadratic			= 4
};

class ActivationFunction
{
public:
	
	ActivationFunction(TipoDiFunzione);
	TipoDiFunzione m_tipo_funzione;

	float CalculateValue(float);
	float CalculateDerivativeValue(float);
};