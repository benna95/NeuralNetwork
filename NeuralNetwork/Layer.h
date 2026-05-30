#pragma once

#include <vector>
#include "FunzioneDiAttivazione.h"
#include "Neurone.h"

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

	std::vector<Neurone> m_neuroni;

	Layer(TipoDiStrato, int, TipoDiFunzione);

	TipoDiStrato m_tipo_di_strato;
	FunzioneDiAttivazione m_funzione;

	float ValutaFunzioneDiAttivazione(float value);
	float ValutaDerivataFunzioneDiAttivazione(float value);
};