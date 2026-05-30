#include "FunzioneDiAttivazione.h"

FunzioneDiAttivazione::FunzioneDiAttivazione(TipoDiFunzione tipo)
{
    switch (tipo)
    {
    case TipoDiFunzione::Lineare:
        m_tipo_funzione = TipoDiFunzione::Lineare;
        break;

    case TipoDiFunzione::Sigmoide:
        m_tipo_funzione = TipoDiFunzione::Sigmoide;
        break;

    case TipoDiFunzione::ReLU:
        m_tipo_funzione = TipoDiFunzione::ReLU;
        break;

    case TipoDiFunzione::TangenteIperbolica:
        m_tipo_funzione = TipoDiFunzione::TangenteIperbolica;
	    break;

    case TipoDiFunzione::Quadratica:
        m_tipo_funzione = TipoDiFunzione::Quadratica;
        break;

    default:
        m_tipo_funzione = TipoDiFunzione::Lineare;
        break;
    }
}

float FunzioneDiAttivazione::CalcolaFunzione(float value)
{
    switch (m_tipo_funzione)
    {
    case TipoDiFunzione::Lineare:
        return value;
        break;

    case TipoDiFunzione::Sigmoide:
        return 1.0f / (1.0f + exp(-value));
        break;

    case TipoDiFunzione::ReLU:
        return value > 0 ? value : 0;
        break;

    case TipoDiFunzione::TangenteIperbolica:
        return tanh(value);
	    break;

    case TipoDiFunzione::Quadratica:
        return value * value;
        break;

    default:
        return 1.0;
        break;
    }
}

float FunzioneDiAttivazione::CalcolaDerivata(float value)
{
    switch (m_tipo_funzione)
    {
    case TipoDiFunzione::Lineare:
        return 1.0f;

    case TipoDiFunzione::Sigmoide:
    {
        float sigmoid = 1.0f / (1.0f + exp(-value));
        return sigmoid * (1.0f - sigmoid);
    }

    case TipoDiFunzione::ReLU:
        return value > 0 ? 1.0f : 0.0f;

    case TipoDiFunzione::TangenteIperbolica:
        return 1.0f - (tanh(value) * tanh(value));
        break;

    case TipoDiFunzione::Quadratica:
        return 2.0 * value;
        break;

    default:
        return 1.0f;
    }
}