#pragma once

#include "ActivationFunction.h"

class Neuron
{
private:
	ActivationFunction m_funzione;
	float m_input;
	float m_output;
	float m_bias;
public:
	Neuron(const ActivationFunction&);

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
