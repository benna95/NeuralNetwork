#include "Layer.h"


Layer::Layer(TipoDiStrato strato, int numero_neuroni, TipoDiFunzione funzione):
m_tipo_di_strato(strato), m_numero_neuroni(numero_neuroni), m_funzione(funzione), 
m_neuroni(numero_neuroni, FunzioneDiAttivazione(funzione))
{

}

float Layer::ValutaFunzioneDiAttivazione(float value)
{
	return m_funzione.CalcolaFunzione(value);
}

float Layer::ValutaDerivataFunzioneDiAttivazione(float value)
{
	return m_funzione.CalcolaDerivata(value);
}