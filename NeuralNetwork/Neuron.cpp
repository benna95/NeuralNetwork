#include "Neuron.h"

Neuron::Neuron(const ActivationFunction &funzione_di_attivazione) : m_funzione(funzione_di_attivazione), m_input(0), m_output(0), m_bias(0)
{
}

void Neuron::SetInput(float value)
{
	m_input = value;
}

void Neuron::SetBias(float value)
{
	m_bias = value;
}

void Neuron::AddInput(float value)
{
	m_input += value;
}

void Neuron::ApplicaFdA()
{
	m_output = m_funzione.CalculateValue(m_input);
}

float Neuron::ApplicaDerivataFdA()
{
	return m_funzione.CalculateDerivativeValue(m_input);
}

float Neuron::GetInput() const
{
	return m_input;
}

float Neuron::GetOutput() const
{
	return m_output;
}

float Neuron::GetBias() const
{
	return m_bias;
}

void Neuron::Reset()
{
	m_input = 0;
	m_output = 0;
}