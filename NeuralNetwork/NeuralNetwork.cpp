#include <iostream>
#include <sstream>
#include <thread>

#include "NeuralNetwork.h"


NeuralNetwork::NeuralNetwork(int m_numero_input, int dimensione_dataset, float learning_rate, int epoche): m_numero_strati(0), 
m_numero_input(m_numero_input), m_dimensione_dataset(dimensione_dataset), x(dimensione_dataset), 
y_target(dimensione_dataset), y_pred(dimensione_dataset), m_numero_epoche(epoche), m_learning_rate(learning_rate), 
m_indici(dimensione_dataset), m_error(epoche), y_pred_originale(m_dimensione_dataset), 
y_target_originale(m_dimensione_dataset) {}

void NeuralNetwork::Add(Layer layer)
{
	Layers.emplace_back(layer);
	m_numero_strati++;
}


void NeuralNetwork::CreaMatriceConnessioni()
{
	m_connessioni = std::vector<std::vector<std::pair<int, int>>>(m_numero_strati - 1);

	std::vector<int> offset_strati(m_numero_strati);

	offset_strati[0] = 1;

	// per ogni strato, salvo l'indice globale del primo neurone
	for (int i = 1; i < m_numero_strati; i++)
	{
		offset_strati[i] = offset_strati[i - 1] + Layers[i - 1].m_neuroni.size();
	}

	for (int i = 0; i < m_numero_strati - 1; i++)
	{
		int numero_neuroni_strato_corrente		= Layers[i].m_neuroni.size();
		int numero_neuroni_strato_successivo	= Layers[i + 1].m_neuroni.size();

		int inizio_strato_corrente		= offset_strati[i];
		int inizio_strato_successivo	= offset_strati[i + 1];

		std::vector<std::pair<int, int>> connessioni_locali;
		connessioni_locali.reserve(numero_neuroni_strato_corrente * numero_neuroni_strato_successivo);

		for (int j = 0; j < numero_neuroni_strato_corrente; j++)
		{
			for (int k = 0; k < numero_neuroni_strato_successivo; k++)
			{
				int indice_neurone_corrente		= inizio_strato_corrente + j;
				int indice_neurone_successivo	= inizio_strato_successivo + k;

				connessioni_locali.push_back({ indice_neurone_corrente, indice_neurone_successivo });
			}
		}

		m_connessioni[i] = connessioni_locali;
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

    for (int i = 0; i < Layers.size(); i++)
    {
		if (Layers[i].m_tipo_di_strato == TipoDiStrato::input || Layers[i].m_tipo_di_strato == TipoDiStrato::nascosto)
		{
			auto numero_neuroni = Layers[i].m_numero_neuroni * Layers[i + 1].m_numero_neuroni;
			m_pesi.emplace_back(numero_neuroni);
			m_bias.emplace_back(numero_neuroni);
			m_numero_pesi.push_back(numero_neuroni);
			m_numero_bias.push_back(numero_neuroni);
			m_gradiente_pesi.emplace_back(numero_neuroni);
			m_gradiente_bias.emplace_back(numero_neuroni);

			for (auto& peso : m_pesi.back())
			{
				//peso = dist(rng);
				peso = dist(rng);
#ifdef _DEBUG
			std::cout << "peso: " << peso << std::endl;
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
        else // Layer::TipoDiStrato::output
        {
            // non fare niente
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

	somma /= m_dimensione_dataset;
	return somma;
}

float NeuralNetwork::CalcolaDerivataLoss()
{
	float somma = 0.0f;

	for (size_t i = 0; i < y_pred.size(); i++)
	{
		somma += 2 * (y_pred[i] - y_target[i]);
	}

	return somma/m_dimensione_dataset;
}

float NeuralNetwork::CalcolaLoss_i(int index)
{
	return (y_pred[index] - y_target[index]) * (y_pred[index] - y_target[index]);
}

float NeuralNetwork::CalcolaDerivataLoss_i(int index)
{
	return 2 * (y_pred[index] - y_target[index]);
}

void NeuralNetwork::LeggiDati(const char* filepath)
{
	std::ifstream file(filepath);

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
	y_target_originale = y_target;
	Shuffle();
}

void NeuralNetwork::BackPropagation(int epoca, int campione_dataset)
{
	int i = campione_dataset;
	auto dLoss_dy_target = CalcolaDerivataLoss_i(i);

		for (size_t j = 0; j < m_gradiente_pesi.size(); j++) 
		{
			for (size_t k = 0; k < m_pesi[j].size(); k++) 
			{
				auto output = Layers[j].m_neuroni[k].GetOutput();
				float gradiente_pesi_locale = output * dLoss_dy_target;
				float gradiente_bias_locale = 1 * dLoss_dy_target;
				for (size_t z = 1 + j; z < m_gradiente_pesi.size(); z++)		
				{
					auto peso = m_pesi[z][k];
					auto derivata_fdA = Layers[z].m_neuroni[k].ApplicaDerivataFdA();

					gradiente_pesi_locale *= peso;
					gradiente_pesi_locale *= derivata_fdA;

					gradiente_bias_locale *= peso;
					gradiente_bias_locale *= derivata_fdA;
				}

				m_gradiente_pesi[j][k] += gradiente_pesi_locale;
				m_gradiente_bias[j][k] += gradiente_bias_locale;
			}

			/*
			for (auto& gradiente_peso : m_gradiente_pesi[j])
			{
				std::cout << "gradiente peso: " << gradiente_peso << std::endl;
			}*/
		}
}

void NeuralNetwork::InizializzaGradienti()
{
	for (size_t i = 0; i < m_gradiente_pesi.size(); i++)
	{
		for (size_t j = 0; j < m_gradiente_pesi[i].size(); j++)
		{
			m_gradiente_pesi[i][j] = 0.0f;
			m_gradiente_bias[i][j] = 0.0f;
		}
	}
}

void NeuralNetwork::ProcessaGradienti()
{
	for (size_t i = 0; i < m_gradiente_pesi.size(); i++)
	{
		for (size_t j = 0; j < m_gradiente_pesi[i].size(); j++)
		{
			m_gradiente_pesi[i][j] /= m_dimensione_dataset;
			m_gradiente_bias[i][j] /= m_dimensione_dataset;
		}
	}
}

void NeuralNetwork::AggiornaPesieBias()
{
	for (int i = 0; i < m_pesi.size(); i++)
	{
		m_pesi[i] = m_pesi[i] - (m_learning_rate * m_gradiente_pesi[i]);
		m_bias[i] = m_bias[i] - (m_learning_rate * m_gradiente_bias[i]);
	}
}

/* full batch gradient descent:
* 1) inizializzo i gradienti a 0
* 2) per ogni campione del dataset faccio inferenza e backpropagation, accumulando i gradienti
* 3) processo i gradienti dividendoli per la dimensione del dataset
* 4) aggiorno pesi e bias usando i gradienti processati
*/

void NeuralNetwork::Train(MetodoDiAddestramento metodo, ChartControl *plot_soluzione, ChartControl *plot_loss_function, std::atomic_bool& QuitRequest)
{
	plot_soluzione->values[0] = GetYOrig();
	plot_soluzione->x_values  = GetX();

	std::vector<float> epoche(m_numero_epoche);
	for (int i = 0; i < epoche.size(); i++)
	{
		epoche[i] = i + 1;
	}

	plot_loss_function->x_values = epoche;

	/*--------------------------CICLO FOR EPICHE--------------------------------------*/
	for (size_t i = 0; i < m_numero_epoche; i++)
	{
		if (QuitRequest)
		{
			return;
		}

#ifdef _DEBUG
		std::cout << "------------------------------" << std::endl;
		std::cout << "epoca: " << i + 1;
#endif // _DEBUG

		InizializzaGradienti();

		for (int j = 0; j < m_dimensione_dataset; j++)
		{
			// Inferenza
			ForwardInference(i, j);

			// BackPropagation
			BackPropagation(i, j);

			if (QuitRequest)
			{
				return;
			}
		}

		ProcessaGradienti();

		// aggiorno pesi
		AggiornaPesieBias();

		m_error[i] = CalcolaLoss();

		
#ifdef _DEBUG
		std::cout << " - errore: " << CalcolaLoss() << std::endl;
		std::cout << "------------------------------" << std::endl;
#endif // _DEBUG

		if (i % 5 == 0 && !QuitRequest)
		{
			if (!gui_update_pending.exchange(true))
			{
				std::lock_guard<std::mutex> lock(m);
#ifdef _DEBUG
				std::cout << "------------------------------" << std::endl;
				std::cout << "lock catturato dal thread: " << std::this_thread::get_id() << std::endl;
#endif // _DEBUG
				RiordinaDati();
#ifdef _DEBUG
				std::cout << "lock rilasciato dal thread: " << std::this_thread::get_id() << std::endl;
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
							std::cout << "lock catturato dal thread principale: " << std::this_thread::get_id() << std::endl;
#endif						
							plot_soluzione->values[1]		= GetYPred();
							plot_loss_function->values[0]	= GetErrorVector();
						}

						plot_soluzione->numero_epoca		= i;
						plot_loss_function->numero_epoca	= i;

						plot_soluzione->Refresh();
						plot_loss_function->Refresh();

#ifdef _DEBUG
						std::cout << "lock rilasciato dal thread principale: " << std::this_thread::get_id() << std::endl;
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

	for (int i = 0; i < m_indici.size(); i++)
		m_indici[i] = i;

	
#ifdef _DEBUG
	std::mt19937 rng(42);
#else
	std::random_device rd;
	std::mt19937 rng(rd());
#endif

	std::shuffle(m_indici.begin(), m_indici.end(), rng);

	std::vector<float> new_x;
	std::vector<float> new_y;

	for (size_t i : m_indici)
	{
		new_x.push_back(x[i]);
		new_y.push_back(y_target[i]);
	}

	x = new_x;
	y_target = new_y;

#ifdef _DEBUG
	std::ofstream debug_file("generate_data/shuffled_data.txt");
	for (int i = 0; i < m_dimensione_dataset; i++)
	{
		debug_file << x[i] << ", " << y_target[i] << std::endl;
	}
	debug_file.close();
#endif // _DEBUG

}

void NeuralNetwork::ForwardInference(int epoca, int campione_dataset)
{
	int i = campione_dataset;
	// ciclo su tutti i layers
	for (int j = 0; j < m_numero_strati; j++)
	{
		if (Layers[j].m_tipo_di_strato == TipoDiStrato::input)
		{
			// sono nello strato di input; quindi l'input è effettivamente "x";
			// ora ho solo un ingresso, ma cmq considero il caso che ho tanti ingressi
			auto numero_ingressi = Layers[j].m_neuroni.size();
			for (size_t k = 0; k < numero_ingressi; k++)
			{
				auto input_value = x[i];
				Layers[j].m_neuroni[k].SetInput(input_value);
				Layers[j].m_neuroni[k].ApplicaFdA(); // applico la FdA, anche se in questo caso è lineare, quindi non cambia nulla
	
				// propagazione
				auto peso = m_pesi[j][k];
				auto bias = m_bias[j][k];
				auto output_current_layer	= Layers[j].m_neuroni[k].GetOutput(); 
				auto output_con_pesi_e_bias = output_current_layer * peso + bias;
				
				// lo mando in ingresso ai neuroni del layer successivo;
				// quanti ne ho nel layer successivo?
				for (size_t z = 0; z < Layers[j + 1].m_neuroni.size(); z++)
				{
					Layers[j + 1].m_neuroni[z].SetInput(output_con_pesi_e_bias);
				}
			}
		}
		else if (Layers[j].m_tipo_di_strato == TipoDiStrato::nascosto)
		{
			auto numero_ingressi = Layers[j].m_neuroni.size();
			for (size_t k = 0; k < numero_ingressi; k++)
			{
				// il neurone ora ha il valore in in ingresso; prima di tutto gli applico la FdA;
				Layers[j].m_neuroni[k].ApplicaFdA();
	
				// ora estraggo il valore;
				auto valore_neurone_con_FdA = Layers[j].m_neuroni[k].GetOutput();
				auto peso = m_pesi[j][k];
				auto bias = m_bias[j][k];
				auto output_con_pesi_e_bias = valore_neurone_con_FdA * peso + bias;
	
				// lo mando in ingresso al neurone del layer successivo
				Layers[j + 1].m_neuroni[k].AddInput(output_con_pesi_e_bias);
			}
		}
		else // Layers[j].m_tipo_di_strato == TipoDiStrato::output
		{
			auto numero_ingressi = Layers[j].m_neuroni.size();
			for (size_t k = 0; k < numero_ingressi; k++)
			{
				Layers[j].m_neuroni[k].ApplicaFdA();
				y_pred[i] = Layers[j].m_neuroni[k].GetOutput();
			}
		}
	}
}

void NeuralNetwork::RiordinaDati()
{
	for (size_t i = 0; i < m_indici.size(); i++)
	{
		int original_index = m_indici[i];
		y_pred_originale[original_index] = y_pred[i];
	}
}

void NeuralNetwork::EsportaDati()
{
	const char* path_output = "generate_data/result.txt";
	std::ofstream output(path_output);

	for (int i = 0; i < m_dimensione_dataset; i++)
	{
		output << x_original[i] << ", " << y_pred_originale[i] << std::endl;
	}
	output.close();

	const char* error_path = "generate_data/error.txt";
	std::ofstream output_error(error_path);

	for (int i = 0; i < m_numero_epoche; i++)
	{
		output_error <<  i << ", " << m_error[i] << std::endl;
	}

	output_error.close();

}

const std::vector<float>& NeuralNetwork::GetYPred() const
{
	return y_pred_originale;
}

const std::vector<float>& NeuralNetwork::GetYOrig() const
{
	return y_target_originale;
}

const std::vector<float>& NeuralNetwork::GetX() const
{
	return x_original;
}

const std::vector<float>& NeuralNetwork::GetErrorVector() const
{
	return m_error;
}
