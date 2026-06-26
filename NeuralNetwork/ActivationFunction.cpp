#include "ActivationFunction.h"

ActivationFunction::ActivationFunction(TipoDiFunzione tipo)
{
    switch (tipo)
    {
    case TipoDiFunzione::Linear:
        m_tipo_funzione = TipoDiFunzione::Linear;
        break;

    case TipoDiFunzione::Sigmoid:
        m_tipo_funzione = TipoDiFunzione::Sigmoid;
        break;

    case TipoDiFunzione::ReLU:
        m_tipo_funzione = TipoDiFunzione::ReLU;
        break;

    case TipoDiFunzione::Tanh:
        m_tipo_funzione = TipoDiFunzione::Tanh;
	    break;

    case TipoDiFunzione::Quadratic:
        m_tipo_funzione = TipoDiFunzione::Quadratic;
        break;

    default:
        m_tipo_funzione = TipoDiFunzione::Linear;
        break;
    }
}

float ActivationFunction::CalculateValue(float value)
{
    switch (m_tipo_funzione)
    {
    case TipoDiFunzione::Linear:
        return value;
        break;

    case TipoDiFunzione::Sigmoid:
        return 1.0f / (1.0f + exp(-value));
        break;

    case TipoDiFunzione::ReLU:
        return value > 0 ? value : 0;
        break;

    case TipoDiFunzione::Tanh:
        return tanh(value);
	    break;

    case TipoDiFunzione::Quadratic:
        return value * value;
        break;

    default:
        return 1.0;
        break;
    }
}

float ActivationFunction::CalculateDerivativeValue(float value)
{
    switch (m_tipo_funzione)
    {
    case TipoDiFunzione::Linear:
        return 1.0f;

    case TipoDiFunzione::Sigmoid:
    {
        float sigmoid = 1.0f / (1.0f + exp(-value));
        return sigmoid * (1.0f - sigmoid);
    }

    case TipoDiFunzione::ReLU:
        return value > 0 ? 1.0f : 0.0f;

    case TipoDiFunzione::Tanh:
        return 1.0f - (tanh(value) * tanh(value));
        break;

    case TipoDiFunzione::Quadratic:
        return 2.0 * value;
        break;

    default:
        return 1.0f;
    }
}