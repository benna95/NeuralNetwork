#include "Layer.h"


Layer::Layer(TipoDiStrato strato, int numero_neuroni, TipoDiFunzione funzione):
m_tipo_di_strato(strato), m_numero_neuroni(numero_neuroni), m_funzione(funzione), 
m_neurons(numero_neuroni, ActivationFunction(funzione))
{

}

float Layer::CalculateActivationFunction(float value)
{
	return m_funzione.CalculateValue(value);
}

float Layer::CalculateDerivativeActivationFunction(float value)
{
	return m_funzione.CalculateDerivativeValue(value);
}