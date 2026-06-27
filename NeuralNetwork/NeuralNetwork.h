#pragma once

#include <fstream>
#include <random>
#include <mutex>
#include <atomic>

#include "Layer.h"
#include "../Graphics/chartcontrol.h"
#include "../Utils/VectorOperations.h"

enum class OptimizationAlgorithm
{
	FullBatchGradientDescent = 0,
	StochasticGradientDescent = 1,
	MinibatchGradientDescent = 2
};

class NeuralNetwork
{
public:
	std::vector<Layer> Layers;
	NeuralNetwork(int, int, float, int);
	int m_layers_number;
	int m_dataset_size;
	int m_inputs_number;
	std::vector<float> x;
	std::vector<float> y_pred;
	std::vector<float> y_target;
	std::vector<float> x_original;
	std::vector<float> m_error;
	std::vector<float> y_target_original;
	std::vector<float> y_pred_original;
	std::vector<int> m_indexes;
	std::vector<std::vector<float>> m_weights;
	std::vector<std::vector<float>> m_weights_gradient;
	std::vector<std::vector<float>> m_bias_gradient;
	std::vector<std::vector<float>> m_bias;
	std::vector<std::vector<float>> m_delta;
	std::vector<int> m_weights_number;
	std::vector<int> m_bias_number;
	float m_learning_rate;
	std::vector<std::vector<std::pair<int, int>>> m_connections_vector;
	// int m_numero_connessioni;

	int epochs_number;

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

	void Train(OptimizationAlgorithm, ChartControl *, ChartControl *, std::atomic_bool &);

	void BackPropagation(int);
	void ForwardInference(int);
	void AggiornaPesieBias();
	void CreaMatriceConnessioni();

	const std::vector<float> &GetYPred() const;
	const std::vector<float> &GetYOrig() const;
	const std::vector<float> &GetX() const;
	const std::vector<float> &GetErrorVector() const;

	void ReadInputData(const char *);
	void EsportaDati();

	float CalcolaLoss();
	float CalcolaDerivataLoss();
	std::mutex m;
	std::atomic<bool> gui_update_pending = false;

	float CalcolaLoss_i(int);
	float CalcolaDerivataLoss_i(int);
};
