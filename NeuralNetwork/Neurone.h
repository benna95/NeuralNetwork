#pragma once

#include "FunzioneDiAttivazione.h"

class Neurone
{
private:
	FunzioneDiAttivazione m_funzione;
	float m_input;
	float m_output;
	float m_bias;
public:
	Neurone(const FunzioneDiAttivazione&);

	void SetInput(float);
	void AddInput(float);
	void ApplicaFdA();
	void Reset();
	void SetBias(float value);

	float GetOutput() const;
	float GetBias() const;
	float ApplicaDerivataFdA();
	float GetInput() const;
};
