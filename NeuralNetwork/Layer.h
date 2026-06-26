#pragma once

#include <vector>
#include "Neuron.h"

enum class TipoDiStrato
{
	input		= 0,
	nascosto	= 1,
	output		= 2
};

class Layer
{
public:
	int m_numero_neuroni;

	std::vector<Neuron> m_neurons;

	Layer(TipoDiStrato, int, TipoDiFunzione);

	TipoDiStrato m_tipo_di_strato;
	ActivationFunction m_funzione;

	float CalculateActivationFunction(float value);
	float CalculateDerivativeActivationFunction(float value);
};