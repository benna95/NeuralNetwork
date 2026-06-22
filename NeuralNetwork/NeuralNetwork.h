#pragma once

#include <fstream>
#include <random>
#include <mutex>
#include <chrono>

#include "Layer.h"
#include "..\Graphics\chartcontrol.h"
#include "..\Utils\VectorOperations.h"

enum class MetodoDiAddestramento
{
	FullBatchGradientDescent	= 0,
	StochasticGradientDescent	= 1,
	MinibatchGradientDescent	= 2
};

class NeuralNetwork
{
public:
	std::vector<Layer> Layers;
	NeuralNetwork(int, int, float, int);
	int m_numero_strati;
	int m_dimensione_dataset;
	int m_numero_input;
	std::vector<float> x;
	std::vector<float> y_pred;
	std::vector<float> y_target;
	std::vector<float> x_original;
	std::vector<float> m_error;
	std::vector<float> y_target_originale;
	std::vector<float> y_pred_originale;
	std::vector<int> m_indici;
	std::vector<std::vector<float>> m_pesi;
	std::vector<std::vector<float>> m_gradiente_pesi;
	std::vector<std::vector<float>> m_gradiente_bias;
	std::vector<std::vector<float>> m_bias;
	std::vector<std::vector<float>> m_delta;
	std::vector<int> m_numero_pesi;
	std::vector<int> m_numero_bias;
	float m_learning_rate;
	std::vector<std::vector<std::pair<int, int>>> m_vettore_connessioni;
	//int m_numero_connessioni;

	int m_numero_epoche;

	void Add(Layer);
	void Shuffle();
#ifdef _DEBUG
	~NeuralNetwork();
#endif // _DEBUG

	

	void InizializzaPesieBias();
	void InizializzaGradienti();
	void ProcessaGradienti();
	void RiordinaDati();
	void ResetNeuroni();

	void Train(MetodoDiAddestramento, ChartControl*, ChartControl*, std::atomic_bool&);

	void BackPropagation(int);
	void ForwardInference(int);
	void AggiornaPesieBias();
	void CreaMatriceConnessioni();

	const std::vector<float>& GetYPred() const;
	const std::vector<float>& GetYOrig() const;
	const std::vector<float>& GetX() const;
	const std::vector<float>& GetErrorVector() const;

	void LeggiDati(const char*);
	void EsportaDati();

	float CalcolaLoss();
	float CalcolaDerivataLoss();
	std::mutex m;
	std::atomic<bool> gui_update_pending = false;

	float CalcolaLoss_i(int);
	float CalcolaDerivataLoss_i(int);
};
