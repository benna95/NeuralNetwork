#pragma once
#include <math.h>
#include <string>

enum class TipoDiFunzione : int
{
	Lineare				= 0,
	ReLU				= 1,
	Sigmoide			= 2,
	TangenteIperbolica	= 3,
	Quadratica			= 4
};

class FunzioneDiAttivazione
{
public:
	
	FunzioneDiAttivazione(TipoDiFunzione);
	TipoDiFunzione m_tipo_funzione;

	float CalcolaFunzione(float);
	float CalcolaDerivata(float);
};