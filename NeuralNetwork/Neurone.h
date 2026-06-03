#pragma once

#include "FunzioneDiAttivazione.h"

class Neurone
{
	FunzioneDiAttivazione m_funzione;
	float m_input;
	float m_output;
public:
	void SetInput(float);
	void AddInput(float);
	float GetInput() const;
	void ApplicaFdA(float);
	float ApplicaDerivataFdA();
	void ApplicaFdA();
	float GetOutput() const;
	Neurone(const FunzioneDiAttivazione &);
	Neurone();
};
