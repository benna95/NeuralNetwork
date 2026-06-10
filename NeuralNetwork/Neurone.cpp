#include "Neurone.h"

Neurone::Neurone(const FunzioneDiAttivazione& funzione_di_attivazione):
	m_funzione(funzione_di_attivazione), m_input(0), m_output(0), m_bias(0)
{
}

void Neurone::SetInput(float value)
{
	m_input = value;
}

void Neurone::SetBias(float value)
{
	m_bias = value;
}

void Neurone::AddInput(float value)
{
	m_input+= value;
}

void Neurone::ApplicaFdA()
{
	m_output = m_funzione.CalcolaFunzione(m_input);
}

float Neurone::ApplicaDerivataFdA()
{
	return m_funzione.CalcolaDerivata(m_input);
}

float Neurone::GetInput() const
{
	return m_input;
}

float Neurone::GetOutput() const
{
	return m_output;
}

float Neurone::GetBias() const
{
	return m_bias;
}

void Neurone::Reset()
{
	m_input = 0;
	m_output = 0;
}