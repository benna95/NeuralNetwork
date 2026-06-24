#include <iostream>
#include <sstream>
#include <thread>

#include "NeuralNetwork.h"


NeuralNetwork::NeuralNetwork(int m_numero_input, int dimensione_dataset, float learning_rate, int epoche): m_layers_number(0), 
m_inputs_number(m_numero_input), m_dataset_size(dimensione_dataset), x(dimensione_dataset), 
y_target(dimensione_dataset), y_pred(dimensione_dataset), epochs_number(epoche), m_learning_rate(learning_rate), 
m_indexes(dimensione_dataset), m_error(epoche), y_pred_original(m_dataset_size), 
y_target_original(m_dataset_size) {}

void NeuralNetwork::Add(Layer layer)
{
	Layers.emplace_back(layer);
	m_layers_number++;
}


void NeuralNetwork::CreaMatriceConnessioni()
{
	m_connections_vector = std::vector<std::vector<std::pair<int, int>>>(m_layers_number - 1);

	// std::vector<int> offset_strati(m_numero_strati);

	// offset_strati[0] = 1;

	// per ogni strato, salvo l'indice globale del primo neurone

	/*
	for (int i = 1; i < m_numero_strati; i++)
	{
		offset_strati[i] = offset_strati[i - 1] + Layers[i - 1].m_neuroni.size();
	}
	*/

	for (int i = 0; i < m_layers_number - 1; i++)
	{
		int numero_neuroni_strato_corrente		= Layers[i].m_neurons.size();
		int numero_neuroni_strato_successivo	= Layers[i + 1].m_neurons.size();

		// int inizio_strato_corrente		= offset_strati[i];
		// int inizio_strato_successivo		= offset_strati[i + 1];

		std::vector<std::pair<int, int>> connessioni_locali;
		connessioni_locali.reserve(numero_neuroni_strato_corrente * numero_neuroni_strato_successivo);

		for (int j = 0; j < numero_neuroni_strato_corrente; j++)
		{
			for (int k = 0; k < numero_neuroni_strato_successivo; k++)
			{
				int indice_neurone_corrente		= j;
				int indice_neurone_successivo	= k;

				connessioni_locali.push_back({ indice_neurone_corrente, indice_neurone_successivo });
			}
		}

		m_connections_vector[i] = connessioni_locali;
	}
}

void NeuralNetwork::InizializzaPesieBias()
{
#ifdef _DEBUG
	std::mt19937 rng(42);
#else
	std::random_device rd;
	std::mt19937 rng(rd());
#endif

    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
	m_delta.resize(Layers.size());

    for (int i = 0; i < Layers.size(); i++)
    {
		m_delta[i].resize(Layers[i].m_neurons.size(), 0.0f);
		if (Layers[i].m_tipo_di_strato == TipoDiStrato::input)
		{
			int numero_pesi = Layers[i].m_numero_neuroni * Layers[i + 1].m_numero_neuroni;
			m_weights.emplace_back(numero_pesi);
			m_weights_number.push_back(numero_pesi);
			m_weights_gradient.emplace_back(numero_pesi);

			for (auto& peso : m_weights.back())
			{
				peso = dist(rng);
#ifdef _DEBUG
				std::cout << "peso: " << peso << std::endl;
#endif // _DEBUG
			}
                
        }
        else if (Layers[i].m_tipo_di_strato == TipoDiStrato::nascosto)
        {
			int numero_pesi = Layers[i].m_numero_neuroni * Layers[i + 1].m_numero_neuroni;
			m_weights.emplace_back(numero_pesi);
			m_weights_number.push_back(numero_pesi);
			m_weights_gradient.emplace_back(numero_pesi);

			int numero_bias = Layers[i].m_numero_neuroni;
			m_bias.emplace_back(numero_bias);
			m_bias_number.push_back(numero_bias);
			m_bias_gradient.emplace_back(numero_bias);

			for (auto& peso : m_weights.back())
			{
				peso = dist(rng);
#ifdef _DEBUG
				std::cout << "weight: " << peso << std::endl;
#endif // _DEBUG
			}
			for (auto& bias : m_bias.back())
			{
				bias = 0;
#ifdef _DEBUG
				std::cout << "bias: " << bias << std::endl;
#endif // _DEBUG
			}
        }
		else // Layers[i].m_tipo_di_strato == TipoDiStrato::output
		{
			int numero_bias = Layers[i].m_numero_neuroni;
			m_bias.emplace_back(numero_bias);
			m_bias_number.push_back(numero_bias);
			m_bias_gradient.emplace_back(numero_bias);

			for (auto& bias : m_bias.back())
			{
				bias = 0;
#ifdef _DEBUG
				std::cout << "bias: " << bias << std::endl;
#endif // _DEBUG
			}
		}
    }
}

void NeuralNetwork::ResetNeuroni()
{
	for (size_t i = 0; i < Layers.size(); i++)
	{
		for (int j = 0; j < Layers[i].m_neurons.size(); j++) 
		{
			Layers[i].m_neurons[j].Reset();
		}
	}
}

float NeuralNetwork::CalcolaLoss()
{
	float somma = 0.0f;

	for (size_t i = 0; i < y_pred.size(); i++)
	{
		somma += (y_pred[i] - y_target[i]) * (y_pred[i] - y_target[i]);
	}

	somma /= m_dataset_size;
	return somma;
}

float NeuralNetwork::CalcolaDerivataLoss()
{
	float somma = 0.0f;

	for (size_t i = 0; i < y_pred.size(); i++)
	{
		somma += 2 * (y_pred[i] - y_target[i]);
	}

	return somma/m_dataset_size;
}

float NeuralNetwork::CalcolaLoss_i(int index)
{
	return (y_pred[index] - y_target[index]) * (y_pred[index] - y_target[index]);
}

float NeuralNetwork::CalcolaDerivataLoss_i(int index)
{
	return 2 * (y_pred[index] - y_target[index]);
}

void NeuralNetwork::ReadInputData(const char* filepath)
{
	std::ifstream file(filepath);
	if (!file)
	{
		wxMessageBox("error to load the file", "error", wxICON_ERROR);
		return;
	}

	std::string line;

	std::string x_value;
	std::string y_value;

	int i = 0;

	if (file)
	{
		while (std::getline(file, line))
		{
			//std::cout << line << std::endl;

			std::stringstream ss(line);

			std::getline(ss, x_value, ',');
			std::getline(ss, y_value, ',');

			x[i]		= std::stof(x_value);
			y_target[i] = std::stof(y_value);
			i++;
		}
	}
	x_original = x;
	y_target_original = y_target;
	Shuffle();
}

void NeuralNetwork::BackPropagation(int campione_dataset)
{
	/*=======================================================================*
	 * RETE 1-2-2-1   -   schema e gradiente del peso w1                   *
	 *                                                                     *
	 *                 H1--w5--H3                                          *
	 *                / \     / \                                          *
	 *            w1 /   \w3 /   \ w7                                      *
	 *              /     \ /     \                                        *
	 *             x       X       Y                                       *
	 *              \     / \     /                                        *
	 *            w2 \   /w4 \   / w8                                      *
	 *                \ /     \ /                                          *
	 *                 H2--w6--H4                                          *
	 *                                                                     *
	 * Pesi:  w1: x->H1    w2: x->H2    w3: H1->H4    w4: H2->H3           *
	 *        w5: H1->H3   w6: H2->H4   w7: H3->Y     w8: H4->Y            *
	 *                                                                     *
	 * --------------------------------------------------------            *
	 * esempio calcolo dLoss / dw1                                         *
	 * --------------------------------------------------------            *
	 *                                                                     *
	 * dLoss/dw1 = Delta_H1 * (input di w1)        // input di w1 = x      *
	 *                                                                     *
	 * Delta_H1 = [ cio' che torna indietro a H1 ] * H1'                   *
	 *            ( H1' = derivata della f. di attivazione di H1 )         *
	 *                                                                     *
	 * Per trovarlo, mi metto in H1 e ripercorro in avanti i cammini       *
	 * fino a Y. Da H1 partono due cammini:  H1->H3->Y  e  H1->H4->Y.      *
	 *                                                                     *
	 * Delta_H1 = 2(Y_pred - Y_target) * [ w5*H3'*w7 + w3*H4'*w8 ] * H1'   *
	 *            \______________/       \____________________/   \__/     *
	 *             delta dell'uscita        somma sui 2 cammini    H1'     *
	 *                                                                     *
	 *=======================================================================*/

	int i = campione_dataset;

	int L = Layers.size() - 1;

	// calcolo delta per i neuroni ultimo strato (output)
	for (int n = 0; n < Layers[L].m_neurons.size(); n++)
	{
		m_delta[L][n] = Layers[L].m_neurons[n].ApplicaDerivataFdA() * CalcolaDerivataLoss_i(i);
	}
	
	// calcolo delta per i neuroni degli strati nascosti, partendo 
	// dall'ultimo strato nascosto e risalendo fino al primo strato nascosto
	for (int j = L - 1; j >= 1; j--)
	{

		for (int n = 0; n < Layers[j].m_neurons.size(); n++)
		{
			float somma = 0.0f;
			for (int c = 0; c < m_connections_vector[j].size(); c++)
			{
				auto [from, to] = m_connections_vector[j][c];
				// sto calcolando il delta del neurone corrente n; questa condizione mi serve  
				// per capire se la connessione c-esima appartiene al neurone che sto considerando, 
				// cioè se arriva al neurone n dello strato subito successivo, ossia neurone n, strato j + 1
				if (from == n)
				{
					// esempio: sto calcolando i contributi di Delta_H1:
					// c = 0 ho 0->0, quindi somma += w5 * Delta_H3 (uguale a w7*H3' * 2(Y_pred - Y_target))
					// c = 1 ho 0->1, quindi somma += w3 * Delta_H4 (uguale a w8*H4' * 2(Y_pred - Y_target))
					// c = 2 ho 1->0, quindi non faccio nulla
					// c = 3 ho 1->1, quindi non faccio nulla
					somma += m_weights[j][c] * m_delta[j + 1][to];
				}
			}
			// ora calcolo effettivamente Delta_H1, ossia moltiplico la somma accumulata * H1';
			m_delta[j][n] = somma * Layers[j].m_neurons[n].ApplicaDerivataFdA();
		}
	}

	// per tutti i neuroni ora conosci Delta_H_i; quindi per calcolare
	// 1) gradiente dei pesi, moltiplico Delta_H_i * input_w_i
	// 2) gradiente dei bias, moltiplico Delta_H_i * 1 (una volta sola per neurone)
	for (int j = 0; j < L; j++)
	{
		for (int c = 0; c < m_connections_vector[j].size(); c++)
		{
			auto [from, to] = m_connections_vector[j][c];
			// accumulo i gradienti per ogni campione del dataset; alla fine della epoca, dividero' per la dimensione del dataset e li resetto
			m_weights_gradient[j][c] += Layers[j].m_neurons[from].GetOutput() * m_delta[j + 1][to];
		}
		for (int n = 0; n < Layers[j + 1].m_neurons.size(); n++)
		{
			m_bias_gradient[j][n] += m_delta[j + 1][n];
		}
	}
}

void NeuralNetwork::InizializzaGradienti()
{
	
	for (size_t i = 0; i < m_weights_gradient.size(); i++)
	{
		for (size_t j = 0; j < m_weights_gradient[i].size(); j++)
		{
			m_weights_gradient[i][j] = 0.0f;
		}
	}

	for (size_t i = 0; i < m_bias_gradient.size(); i++)
	{
		for (size_t j = 0; j < m_bias_gradient[i].size(); j++)
		{
			m_bias_gradient[i][j] = 0.0f;
		}
	}
}

void NeuralNetwork::ProcessaGradienti()
{
	
	for (size_t i = 0; i < m_weights_gradient.size(); i++)
	{
		for (size_t j = 0; j < m_weights_gradient[i].size(); j++)
		{
			m_weights_gradient[i][j] /= m_dataset_size;
		}
	}

	for (size_t i = 0; i < m_bias_gradient.size(); i++)
	{
		for (size_t j = 0; j < m_bias_gradient[i].size(); j++)
		{
			m_bias_gradient[i][j] /= m_dataset_size;
		}
	}
	
	
}

void NeuralNetwork::AggiornaPesieBias()
{
	for (int i = 0; i < m_weights.size(); i++)
	{
		m_weights[i] = m_weights[i] - (m_learning_rate * m_weights_gradient[i]);
	}

	for (int i = 0; i < m_bias.size(); i++)
	{
		m_bias[i] = m_bias[i] - (m_learning_rate * m_bias_gradient[i]);
	}

	for (int i = 1; i < Layers.size(); i++)
	{
		for (size_t j = 0; j < Layers[i].m_numero_neuroni; j++)
		{
			Layers[i].m_neurons[j].SetBias(m_bias[i - 1][j]);
		}
	}
}

/* full batch gradient descent:
* 1) inizializzo i gradienti a 0
* 2) per ogni campione del dataset faccio inferenza e backpropagation, accumulando i gradienti
* 3) processo i gradienti dividendoli per la dimensione del dataset
* 4) aggiorno pesi e bias usando i gradienti processati
*/

void NeuralNetwork::Train(OptimizationAlgorithm metodo, ChartControl *plot_soluzione, ChartControl *plot_loss_function, std::atomic_bool& QuitRequest)
{
	plot_soluzione->values[0] = GetYOrig();
	plot_soluzione->x_values  = GetX();

	std::vector<float> epoche(epochs_number);
	for (int i = 0; i < epoche.size(); i++)
	{
		epoche[i] = i + 1;
	}

	plot_loss_function->x_values = epoche;

	/*--------------------------CICLO FOR EPICHE--------------------------------------*/
	for (size_t i = 0; i < epochs_number; i++)
	{
		if (QuitRequest)
		{
			return;
		}

#ifdef _DEBUG
		std::cout << "------------------------------" << std::endl;
		std::cout << "epoch: " << i + 1;
#endif // _DEBUG

		InizializzaGradienti();

		for (int j = 0; j < m_dataset_size; j++)
		{
			// Inferenza
			ForwardInference(j);
			
			// BackPropagation
			BackPropagation(j);

			if (QuitRequest)
			{
				return;
			}

			//ResetNeuroni();
		}

		ProcessaGradienti();

		// aggiorno pesi
		AggiornaPesieBias();

		m_error[i] = CalcolaLoss();

#ifdef _DEBUG
		std::cout << " - error: " << m_error[i] << std::endl;
		std::cout << "------------------------------" << std::endl;
#endif // _DEBUG

		if (i % 5 == 0 && !QuitRequest)
		{
			if (!gui_update_pending.exchange(true))
			{
				std::lock_guard<std::mutex> lock(m);
#ifdef _DEBUG
				std::cout << "------------------------------" << std::endl;
				std::cout << "lock catched by thread: " << std::this_thread::get_id() << std::endl;
#endif // _DEBUG
				RiordinaDati();
#ifdef _DEBUG
				std::cout << "lock released by thread: " << std::this_thread::get_id() << std::endl;
				std::cout << "------------------------------" << std::endl;
#endif // _DEBUG

				plot_soluzione->GetParent()->CallAfter([plot_soluzione, plot_loss_function, i, this, &QuitRequest]()
					{
						// la lambda potrebbe essere eseguita dopo che ho richiesto la distruzione della finestra;
						// CallAfter non esegue subito la lambda, la mette in coda negli eventi della GUI
						if (QuitRequest)
						{
							gui_update_pending = false;
							return;
						}

						{
							std::lock_guard<std::mutex> lock(this->m);
#ifdef _DEBUG
							std::cout << "------------------------------" << std::endl;
							std::cout << "lock catched by the main thread: " << std::this_thread::get_id() << std::endl;
#endif						
							plot_soluzione->values[1]		= GetYPred();
							plot_loss_function->values[0]	= GetErrorVector();
						}

						plot_soluzione->numero_epoca		= i;
						plot_loss_function->numero_epoca	= i;

						plot_soluzione->Refresh();
						plot_loss_function->Refresh();

#ifdef _DEBUG
						std::cout << "lock released by the main thread: " << std::this_thread::get_id() << std::endl;
						std::cout << "------------------------------" << std::endl;
#endif
						gui_update_pending = false;
					});
			}
		}
	}/*--------------------------CICLO FOR EPICHE--------------------------------------*/
}

void NeuralNetwork::Shuffle()
{

	for (int i = 0; i < m_indexes.size(); i++)
		m_indexes[i] = i;

	
#ifdef _DEBUG
	std::mt19937 rng(42);
#else
	std::random_device rd;
	std::mt19937 rng(rd());
#endif

	std::shuffle(m_indexes.begin(), m_indexes.end(), rng);

	std::vector<float> new_x;
	std::vector<float> new_y;

	for (size_t i : m_indexes)
	{
		new_x.push_back(x[i]);
		new_y.push_back(y_target[i]);
	}

	x = new_x;
	y_target = new_y;

#ifdef _DEBUG
	std::ofstream debug_file("generate_data/shuffled_data.txt");
	for (int i = 0; i < m_dataset_size; i++)
	{
		debug_file << x[i] << ", " << y_target[i] << std::endl;
	}
	debug_file.close();
#endif // _DEBUG

}

void NeuralNetwork::ForwardInference(int campione_dataset)
{
	int i = campione_dataset;

	// setto input
	for (size_t j = 0; j < Layers[0].m_neurons.size(); j++)
	{
		Layers[0].m_neurons[j].SetInput(x[i]);
		Layers[0].m_neurons[j].ApplicaFdA(); // è lineare quindi non cambia nulla
	}

	// ciclo sui layer fino ad arrivare al penultimo
	for (size_t layer = 0; layer < m_connections_vector.size(); layer++)
	{
		// ciclo sui neuroni dello strato successivo. questo ciclo è parallelizzabile, 
		// ogni neurone dello strato successivo può essere calcolato indipendentemente dagli altri
		for (size_t n = 0; n < Layers[layer + 1].m_neurons.size(); n++)
		{
			float somma = 0.0f;

			// ciclo su tutte le connessioni che arrivano al neurone n dello strato layer + 1.
			for (size_t c = 0; c < m_connections_vector[layer].size(); c++)
			{

				auto [from, to] = m_connections_vector[layer][c];

				// se la connessione arriva al neurone n dello strato layer + 1, 
				// allora prendo il contributo del neurone "from" dello strato layer
				if (to == n)
				{
					float output_from = Layers[layer].m_neurons[from].GetOutput();
					float peso = m_weights[layer][c];

					// sto sommando tutti i contributi che arrivano al neurone n dello strato layer + 1
					somma += output_from * peso;
				}
			}

			// aggiungo il bias
			somma += Layers[layer + 1].m_neurons[n].GetBias();

			// imposto l'input totale al neurone n dello strato layer + 1 e applico la funzione di attivazione
			Layers[layer + 1].m_neurons[n].SetInput(somma);
			Layers[layer + 1].m_neurons[n].ApplicaFdA();
		}
	}

	// calcolo y_pred
	y_pred[i] = Layers[m_layers_number - 1].m_neurons[0].GetOutput();
}

void NeuralNetwork::RiordinaDati()
{
	for (size_t i = 0; i < m_indexes.size(); i++)
	{
		int original_index = m_indexes[i];
		y_pred_original[original_index] = y_pred[i];
	}
}

void NeuralNetwork::EsportaDati()
{
	const char* path_output = "generate_data/result.txt";
	std::ofstream output(path_output);

	for (int i = 0; i < m_dataset_size; i++)
	{
		output << x_original[i] << ", " << y_pred_original[i] << std::endl;
	}
	output.close();

	const char* error_path = "generate_data/error.txt";
	std::ofstream output_error(error_path);

	for (int i = 0; i < epochs_number; i++)
	{
		output_error <<  i << ", " << m_error[i] << std::endl;
	}

	output_error.close();

}

const std::vector<float>& NeuralNetwork::GetYPred() const
{
	return y_pred_original;
}

const std::vector<float>& NeuralNetwork::GetYOrig() const
{
	return y_target_original;
}

const std::vector<float>& NeuralNetwork::GetX() const
{
	return x_original;
}

const std::vector<float>& NeuralNetwork::GetErrorVector() const
{
	return m_error;
}

#ifdef _DEBUG
NeuralNetwork::~NeuralNetwork()
{
	std::cout << "deleting Network: " << this << std::endl;
}

#endif // _DEBUG

