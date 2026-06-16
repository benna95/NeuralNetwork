#include "Frame.h"

wxDEFINE_EVENT(EVT_PARAMETRI_VALIDI, wxCommandEvent);

MyFrame::MyFrame(const wxString& title, const wxPoint& pos, const wxSize& size)
    : wxFrame(NULL, wxID_ANY, title, pos, size)
{
    //auto main_sizer = new wxBoxSizer(wxVERTICAL);

    /*

    plot_soluzione = new ChartControl(
        this,
        wxID_ANY,
        wxDefaultPosition,
        wxDefaultSize,
        TipodiGrafico::soluzione
    );

    plot_soluzione->title = "Neural Network";

    plot_loss_function = new ChartControl(
        this,
        wxID_ANY,
        wxDefaultPosition,
        wxDefaultSize,
        TipodiGrafico::loss_function
    );
    */

    //plot_loss_function->title = "Loss Function";

    this->Bind(wxEVT_CLOSE_WINDOW, &MyFrame::OnClose, this);

#ifdef _DEBUG
    this->Bind(wxEVT_SIZE, &MyFrame::PrintWindowsSize, this);
#endif

    //main_sizer->Add(plot_soluzione, 2, wxEXPAND | wxALL, FromDIP(5));
    //main_sizer->Add(plot_loss_function, 1, wxEXPAND | wxALL, FromDIP(5));

    //SetSizer(main_sizer);
    //SetMinClientSize(size);

    CreateWindow();

	SetUpRete->Bind(wxEVT_BUTTON, &MyFrame::SetUpReteNeurale, this);


    m_text_ctrl_numero_input->Bind(wxEVT_TEXT, &MyFrame::CheckParametri, this);
    m_text_ctrl_strati_nascosti->Bind(wxEVT_TEXT, &MyFrame::CheckParametri, this);
    m_text_ctrl_neuroni_per_strato_nascosto->Bind(wxEVT_TEXT, &MyFrame::CheckParametri, this);
    m_text_ctrl_output->Bind(wxEVT_TEXT, &MyFrame::CheckParametri, this);

    Bind(EVT_PARAMETRI_VALIDI, &MyFrame::AggiornaParametri, this);

    m_choice_strati_nascosti->Bind(wxEVT_CHOICE, &MyFrame::OnSceltaFdAStratiNascosti, this);
    m_choice_output_activation->Bind(wxEVT_CHOICE, &MyFrame::OnSceltaFdAOutput, this);

    //StartTraining();
}

void MyFrame::StartTraining()
{
    if (!isProcessing)
    {
        isProcessing = true;

        auto training_function = [this]()
            {
                Network->Train(
                    MetodoDiAddestramento::FullBatchGradientDescent,
                    plot_soluzione,
                    plot_loss_function,
                    QuitRequest
                );

                if (QuitRequest)
                {
                    this->CallAfter([this]()
                        {
                            isProcessing = false;
                            background_thread.join();
                            Destroy();
                        });
                }
                else
                {
                    this->CallAfter([this]()
                        {
                            background_thread.join();
                            isProcessing = false;
                        });
                }
            };

        background_thread = std::thread(training_function);
    }
}

void MyFrame::CreateWindow()
{
    wxBoxSizer* main_vertical_sizer = new wxBoxSizer(wxVERTICAL);

    /* ============================ ARCHITETTURA ============================ */

    auto pannello_architettura = new wxPanel(this, wxID_ANY);
    wxBoxSizer* sizer_pannello_architettura = new wxBoxSizer(wxVERTICAL);

    // linea + titolo di sezione
    auto staticline1 = new wxStaticLine(pannello_architettura, wxID_ANY);
    sizer_pannello_architettura->Add(staticline1, 0, wxEXPAND | wxALL, 5);

    auto m_static_text_architettura = new wxStaticText(pannello_architettura, wxID_ANY, _("Architettura"));
    m_static_text_architettura->SetFont(wxFont(14, wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD));
    sizer_pannello_architettura->Add(m_static_text_architettura, 0, wxALL | wxALIGN_CENTER_HORIZONTAL, 5);

    // riga: numero input
    auto m_panel_num_input = new wxPanel(pannello_architettura, wxID_ANY);
    wxBoxSizer* sizer_panel_num_input = new wxBoxSizer(wxHORIZONTAL);

    auto m_static_text_numero_input = new wxStaticText(m_panel_num_input, wxID_ANY, _("numero input"));
    m_text_ctrl_numero_input = new wxTextCtrl(m_panel_num_input, wxID_ANY, wxEmptyString);
    m_text_ctrl_numero_input->SetValidator(wxTextValidator(wxFILTER_DIGITS, &m_numero_input));

    sizer_panel_num_input->Add(m_static_text_numero_input, 0, wxALL, 5);
    sizer_panel_num_input->AddStretchSpacer(1);
    sizer_panel_num_input->Add(m_text_ctrl_numero_input, 0, wxALL, 5);

    m_panel_num_input->SetSizer(sizer_panel_num_input);
    sizer_pannello_architettura->Add(m_panel_num_input, 0, wxEXPAND | wxALL, 5);

    // riga: strati nascosti (+ funzione di attivazione)
    auto m_panel_strati_nascosti = new wxPanel(pannello_architettura, wxID_ANY);
    wxBoxSizer* sizer_panel_strati_nascosti = new wxBoxSizer(wxHORIZONTAL);

    auto m_static_text_strati_nascosti = new wxStaticText(m_panel_strati_nascosti, wxID_ANY, _("strati nascosti"));
    m_text_ctrl_strati_nascosti = new wxTextCtrl(m_panel_strati_nascosti, wxID_ANY, wxEmptyString);
    m_text_ctrl_strati_nascosti->SetValidator(wxTextValidator(wxFILTER_DIGITS, &m_numero_strati_nascosti));


    wxArrayString m_choice_strati_nascosti_choices;
    m_choice_strati_nascosti_choices.Add(_("Lineare"));
    m_choice_strati_nascosti_choices.Add(_("ReLU"));
    m_choice_strati_nascosti_choices.Add(_("Sigmoid"));
    m_choice_strati_nascosti_choices.Add(_("Tanh"));
    m_choice_strati_nascosti_choices.Add(_("Quadratica"));

    m_choice_strati_nascosti = new wxChoice(m_panel_strati_nascosti, wxID_ANY, wxDefaultPosition, wxDefaultSize, m_choice_strati_nascosti_choices);
    m_choice_strati_nascosti->SetSelection(0);

    sizer_panel_strati_nascosti->Add(m_static_text_strati_nascosti, 0, wxALL, 5);
    sizer_panel_strati_nascosti->AddStretchSpacer(1);
    sizer_panel_strati_nascosti->Add(m_text_ctrl_strati_nascosti, 0, wxALL, 5);
    sizer_panel_strati_nascosti->Add(m_choice_strati_nascosti, 0, wxALL, 5);

    m_panel_strati_nascosti->SetSizer(sizer_panel_strati_nascosti);
    sizer_pannello_architettura->Add(m_panel_strati_nascosti, 0, wxEXPAND | wxALL, 5);


    // riga: numero neuroni x strato nascosto
	auto panel_neuroni_per_strato_nascosto = new wxPanel(pannello_architettura, wxID_ANY);
    auto m_static_text_neuroni_per_strato_nascosto = new wxStaticText(panel_neuroni_per_strato_nascosto, wxID_ANY, _("neuroni per strato nascosto"));
    m_text_ctrl_neuroni_per_strato_nascosto = new wxTextCtrl(panel_neuroni_per_strato_nascosto, wxID_ANY, wxEmptyString);
    m_text_ctrl_neuroni_per_strato_nascosto->SetValidator(wxTextValidator(wxFILTER_DIGITS, &m_numero_neuroni_per_strato_nascosto));

    wxBoxSizer* sizer_panel_neuroni_per_strato_nascosto = new wxBoxSizer(wxHORIZONTAL);
    sizer_panel_neuroni_per_strato_nascosto->Add(m_static_text_neuroni_per_strato_nascosto, 0, wxALL, 5);
    sizer_panel_neuroni_per_strato_nascosto->AddStretchSpacer(1);
    sizer_panel_neuroni_per_strato_nascosto->Add(m_text_ctrl_neuroni_per_strato_nascosto, 0, wxALL, 5);
    panel_neuroni_per_strato_nascosto->SetSizer(sizer_panel_neuroni_per_strato_nascosto);

    sizer_pannello_architettura->Add(panel_neuroni_per_strato_nascosto, 0, wxEXPAND | wxALL, 5);
    

    // riga: output (+ funzione di attivazione)
    auto m_panel_output = new wxPanel(pannello_architettura, wxID_ANY);
    wxBoxSizer* sizer_panel_output = new wxBoxSizer(wxHORIZONTAL);

    auto m_static_text_output = new wxStaticText(m_panel_output, wxID_ANY, _("output"));
    m_text_ctrl_output = new wxTextCtrl(m_panel_output, wxID_ANY, wxEmptyString);
    m_text_ctrl_output->SetValidator(wxTextValidator(wxFILTER_DIGITS, &m_numero_output));

    wxArrayString m_choice_output_choices;
    //m_choice_output_choices.Add(_("Softmax"));
    //m_choice_output_choices.Add(_("Sigmoid"));
    m_choice_output_choices.Add(_("Lineare"));
    m_choice_output_activation = new wxChoice(m_panel_output, wxID_ANY, wxDefaultPosition, wxDefaultSize, m_choice_output_choices);
    m_choice_output_activation->SetSelection(0);

    sizer_panel_output->Add(m_static_text_output, 0, wxALL, 5);
    sizer_panel_output->AddStretchSpacer(1);
    sizer_panel_output->Add(m_text_ctrl_output, 0, wxALL, 5);
    sizer_panel_output->Add(m_choice_output_activation, 0, wxALL, 5);

    m_panel_output->SetSizer(sizer_panel_output);
    sizer_pannello_architettura->Add(m_panel_output, 0, wxEXPAND | wxALL, 5);

    // linea di chiusura sezione
    auto staticline2 = new wxStaticLine(pannello_architettura, wxID_ANY);
    sizer_pannello_architettura->Add(staticline2, 0, wxEXPAND | wxALL, 5);

    pannello_architettura->SetSizer(sizer_pannello_architettura);
    main_vertical_sizer->Add(pannello_architettura, 0, wxEXPAND | wxALL, 5);


    /* ============================ RIASSUNTO RETE ============================ */

    auto pannello_parametri = new wxPanel(this, wxID_ANY);
    wxBoxSizer* sizer_pannello_parametri = new wxBoxSizer(wxVERTICAL);

    // linea + titolo di sezione
    auto staticline3 = new wxStaticLine(pannello_parametri, wxID_ANY);
    sizer_pannello_parametri->Add(staticline3, 0, wxEXPAND | wxALL, 5);

    auto m_static_text_riassunto_rete = new wxStaticText(pannello_parametri, wxID_ANY, _("Riassunto Rete"));
    m_static_text_riassunto_rete->SetFont(wxFont(14, wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD));
    sizer_pannello_parametri->Add(m_static_text_riassunto_rete, 0, wxALL | wxALIGN_CENTER_HORIZONTAL, 5);

    // riga centrata: pesi / bias / totali
    auto m_panel_parametri_riassunto = new wxPanel(pannello_parametri, wxID_ANY);
    wxBoxSizer* sizer_parametri_riassunto = new wxBoxSizer(wxHORIZONTAL);

    m_static_text_numero_pesi       = new wxStaticText(m_panel_parametri_riassunto, wxID_ANY, _("numero pesi: "));
    m_static_text_numero_bias       = new wxStaticText(m_panel_parametri_riassunto, wxID_ANY, _("numero bias:"));
    m_static_text_parametri_totali  = new wxStaticText(m_panel_parametri_riassunto, wxID_ANY, _("parametri totali:"));

    sizer_parametri_riassunto->AddStretchSpacer(1);
    sizer_parametri_riassunto->Add(m_static_text_numero_pesi, 0, wxALL, 5);
    sizer_parametri_riassunto->Add(m_static_text_numero_bias, 0, wxALL, 5);
    sizer_parametri_riassunto->Add(m_static_text_parametri_totali, 0, wxALL, 5);
    sizer_parametri_riassunto->AddStretchSpacer(1);

    m_panel_parametri_riassunto->SetSizer(sizer_parametri_riassunto);
    sizer_pannello_parametri->Add(m_panel_parametri_riassunto, 0, wxEXPAND | wxALL, 5);

    // linea di chiusura sezione
    auto staticline4 = new wxStaticLine(pannello_parametri, wxID_ANY);
    sizer_pannello_parametri->Add(staticline4, 0, wxEXPAND | wxALL, 5);

    pannello_parametri->SetSizer(sizer_pannello_parametri);
    main_vertical_sizer->Add(pannello_parametri, 0, wxEXPAND | wxALL, 5);


    /* ============================ TRAINING ============================ */

    auto pannello_training = new wxPanel(this, wxID_ANY);
    wxBoxSizer* sizer_pannello_training = new wxBoxSizer(wxVERTICAL);

    // linea + titolo di sezione
    auto m_staticline_training = new wxStaticLine(pannello_training, wxID_ANY);
    sizer_pannello_training->Add(m_staticline_training, 0, wxEXPAND | wxALL, 5);

    auto m_static_text_training = new wxStaticText(pannello_training, wxID_ANY, _("Training"));
    m_static_text_training->SetFont(wxFont(14, wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD));
    sizer_pannello_training->Add(m_static_text_training, 0, wxALL | wxALIGN_CENTER_HORIZONTAL, 5);

    // riga: tipo di metodo
    auto m_panel_tipo_metodo = new wxPanel(pannello_training, wxID_ANY);
    wxBoxSizer* sizer_panel_tipo_metodo = new wxBoxSizer(wxHORIZONTAL);

    auto m_static_text_tipo_di_metodo = new wxStaticText(m_panel_tipo_metodo, wxID_ANY, _("tipo di metodo"));

    wxArrayString m_choice_tipo_di_metodo_choices;
    m_choice_tipo_di_metodo_choices.Add(_("Full batch gradient descent"));
    m_choice_tipo_di_metodo = new wxChoice(m_panel_tipo_metodo, wxID_ANY, wxDefaultPosition, wxDefaultSize, m_choice_tipo_di_metodo_choices);
    m_choice_tipo_di_metodo->SetSelection(0);

    sizer_panel_tipo_metodo->Add(m_static_text_tipo_di_metodo, 0, wxALL, 5);
    sizer_panel_tipo_metodo->AddStretchSpacer(1);
    sizer_panel_tipo_metodo->Add(m_choice_tipo_di_metodo, 0, wxALL, 5);

    m_panel_tipo_metodo->SetSizer(sizer_panel_tipo_metodo);
    sizer_pannello_training->Add(m_panel_tipo_metodo, 0, wxEXPAND | wxALL, 5);

    // riga: numero epoche
    auto m_panel_numero_epoche = new wxPanel(pannello_training, wxID_ANY);
    wxBoxSizer* sizer_panel_numero_epoche = new wxBoxSizer(wxHORIZONTAL);

    auto m_static_text_numero_epoche = new wxStaticText(m_panel_numero_epoche, wxID_ANY, _("numero epoche"));
    m_text_ctrl_numero_epoche = new wxTextCtrl(m_panel_numero_epoche, wxID_ANY, wxEmptyString);
	m_text_ctrl_numero_epoche->SetValidator(wxTextValidator(wxFILTER_DIGITS, &m_numero_epoche));

    sizer_panel_numero_epoche->Add(m_static_text_numero_epoche, 0, wxALL, 5);
    sizer_panel_numero_epoche->AddStretchSpacer(1);
    sizer_panel_numero_epoche->Add(m_text_ctrl_numero_epoche, 0, wxALL, 5);

    m_panel_numero_epoche->SetSizer(sizer_panel_numero_epoche);
    sizer_pannello_training->Add(m_panel_numero_epoche, 0, wxEXPAND | wxALL, 5);

    // riga: learning rate
    auto m_panel_learning_rate = new wxPanel(pannello_training, wxID_ANY);
    wxBoxSizer* sizer_panel_learning_rate = new wxBoxSizer(wxHORIZONTAL);

    auto m_static_text_learning_rate = new wxStaticText(m_panel_learning_rate, wxID_ANY, _("learning rate"));
    m_text_ctrl_learning_rate = new wxTextCtrl(m_panel_learning_rate, wxID_ANY, wxEmptyString);
    m_text_ctrl_learning_rate->SetValidator(wxTextValidator(wxFILTER_NUMERIC, &m_learning_rate));

    sizer_panel_learning_rate->Add(m_static_text_learning_rate, 0, wxALL, 5);
    sizer_panel_learning_rate->AddStretchSpacer(1);
    sizer_panel_learning_rate->Add(m_text_ctrl_learning_rate, 0, wxALL, 5);

    m_panel_learning_rate->SetSizer(sizer_panel_learning_rate);
    sizer_pannello_training->Add(m_panel_learning_rate, 0, wxEXPAND | wxALL, 5);

    // riga: parallelizzazione
    auto m_panel_parallelizazione = new wxPanel(pannello_training, wxID_ANY);
    wxBoxSizer* sizer_panel_parallelizazione = new wxBoxSizer(wxHORIZONTAL);

    auto m_static_text_parallelizzazione = new wxStaticText(m_panel_parallelizazione, wxID_ANY, _("parallelizazione"));

    wxArrayString m_choice_parallelizzazione_choices;
    m_choice_parallelizzazione_choices.Add(_("Seriale"));
    m_choice_parallelizzazione_choices.Add(_("OpenMP"));
    m_choice_parallelizzazione = new wxChoice(m_panel_parallelizazione, wxID_ANY, wxDefaultPosition, wxDefaultSize, m_choice_parallelizzazione_choices);
    m_choice_parallelizzazione->SetSelection(0);

    sizer_panel_parallelizazione->Add(m_static_text_parallelizzazione, 0, wxALL, 5);
    sizer_panel_parallelizazione->AddStretchSpacer(1);
    sizer_panel_parallelizazione->Add(m_choice_parallelizzazione, 0, wxALL, 5);

    m_panel_parallelizazione->SetSizer(sizer_panel_parallelizazione);
    sizer_pannello_training->Add(m_panel_parallelizazione, 0, wxEXPAND | wxALL, 5);

    auto m_staticline_training2 = new wxStaticLine(pannello_training, wxID_ANY);
    sizer_pannello_training->Add(m_staticline_training2, 0, wxEXPAND | wxALL, 5);

    pannello_training->SetSizer(sizer_pannello_training);

    

    main_vertical_sizer->Add(pannello_training, 0, wxEXPAND | wxALL, 5);


    /* ============================ BOTTONI ============================ */

    auto pannello_bottoni = new wxPanel(this, wxID_ANY);
    wxBoxSizer* sizer_pannello_bottoni = new wxBoxSizer(wxHORIZONTAL);

    SetUpRete = new wxButton(pannello_bottoni, wxID_ANY, _("SetUp rete"));
    //AvviaRete->Disable();
    AvviaRete = new wxButton(pannello_bottoni, wxID_ANY, _("Avvia rete"));
    AvviaRete->Disable();   // si abilita dopo il setup

    sizer_pannello_bottoni->AddStretchSpacer(1);
    sizer_pannello_bottoni->Add(SetUpRete, 0, wxALL, 5);
    sizer_pannello_bottoni->Add(AvviaRete, 0, wxALL, 5);
    sizer_pannello_bottoni->AddStretchSpacer(1);

    pannello_bottoni->SetSizer(sizer_pannello_bottoni);
    main_vertical_sizer->Add(pannello_bottoni, 0, wxEXPAND | wxALL, 5);

    this->SetSizerAndFit(main_vertical_sizer);
}

void MyFrame::OnClose(wxCloseEvent& event)
{
#ifdef _DEBUG
    std::cout << "sto uscendo dall'applicazione\n";
#endif

    if (isProcessing)
    {
        event.Veto();
        QuitRequest = true;
    }
    else
    {
        this->Destroy();
    }
}

MyFrame::~MyFrame()
{
    delete Network;
}

#ifdef _DEBUG
void MyFrame::PrintWindowsSize(wxSizeEvent& event)
{
    std::cout << "WIDTH: " << this->GetSize().GetWidth() << std::endl;
    std::cout << "HEIGHT: " << this->GetSize().GetHeight() << std::endl;

    event.Skip();
}
#endif

void MyFrame::SetUpReteNeurale(wxCommandEvent& event)
{
	TransferDataFromWindow();

    Network = new NeuralNetwork(
    std::stoi(m_numero_input.ToStdString()),    // numero_input
    1000,                                       // campione_dataset
    std::stof(m_learning_rate.ToStdString()),   // learning_rate
    std::stoi(m_numero_epoche.ToStdString())    // numero_epoche
    ); 

    Network->LeggiDati("generate_data/data.txt");

    Network->Add({ TipoDiStrato::input,    1, TipoDiFunzione::Lineare });

    /*
        Lineare             = 0,
        ReLU                = 1,
        Sigmoide            = 2,
        TangenteIperbolica  = 3,
        Quadratica          = 4
    */

    for (int i = 0; i < std::stoi(m_numero_strati_nascosti.ToStdString()); i++)
    {
        Network->Add({ TipoDiStrato::nascosto, std::stoi(m_numero_neuroni_per_strato_nascosto.ToStdString()), m_tipo_funzione_strati_nascosti});
    }

    Network->Add({ TipoDiStrato::output,   /*std::stoi(m_numero_output.ToStdString())*/ 1, m_tipo_funzione_output});

    Network->InizializzaPesieBias();
    Network->CreaMatriceConnessioni();

	AvviaRete->Enable();
}

void MyFrame::AggiornaParametri(wxCommandEvent& event)
{
    TransferDataFromWindow();

    // se anche uno solo dei campi è vuoto, non procedo oltre
    if (m_text_ctrl_numero_input->IsEmpty() || m_text_ctrl_output->IsEmpty() || 
    m_text_ctrl_neuroni_per_strato_nascosto->IsEmpty() || m_text_ctrl_strati_nascosti->IsEmpty())
    {
        return;
    }

    int numero_input        = std::stoi(m_numero_input.ToStdString());
    int strati_nascosti     = std::stoi(m_numero_strati_nascosti.ToStdString());
    int neuroni_per_strato  = std::stoi(m_numero_neuroni_per_strato_nascosto.ToStdString());
    int numero_output       = std::stoi(m_numero_output.ToStdString());

    int numero_totale_pesi = 0;
    int numero_totale_bias = 0;

    numero_totale_pesi += numero_input * neuroni_per_strato;                                // input -> primo strato nascosto
    numero_totale_pesi += neuroni_per_strato * neuroni_per_strato * (strati_nascosti - 1);  // strati nascosti intermedi
    numero_totale_pesi += neuroni_per_strato * numero_output;                               // ultimo strato nascosto -> output

    numero_totale_bias += neuroni_per_strato * strati_nascosti;                             // bias di tutti gli strati nascosti
    numero_totale_bias += numero_output;                                                    // bias dello strato output

    m_static_text_numero_pesi->SetLabel(
        wxString::Format("numero pesi: %d", numero_totale_pesi));
    m_static_text_numero_bias->SetLabel(
        wxString::Format("numero bias: %d", numero_totale_bias));
    m_static_text_parametri_totali->SetLabel(
        wxString::Format("parametri totali: %d", numero_totale_pesi + numero_totale_bias));

    this->Layout();
}

void MyFrame::CheckParametri(wxCommandEvent& event)
{
    if (m_text_ctrl_numero_input->Validate() && m_text_ctrl_strati_nascosti->Validate() &&
        m_text_ctrl_neuroni_per_strato_nascosto->Validate() && m_text_ctrl_output->Validate())
        {
            wxCommandEvent evt(EVT_PARAMETRI_VALIDI);
            wxPostEvent(this, evt);
        }

    event.Skip();
}

void MyFrame::OnSceltaFdAStratiNascosti(wxCommandEvent&)
{
    m_tipo_funzione_strati_nascosti = (TipoDiFunzione)m_choice_strati_nascosti->GetSelection();
}

void MyFrame::OnSceltaFdAOutput(wxCommandEvent&)
{
    m_tipo_funzione_output = (TipoDiFunzione)m_choice_output_activation->GetSelection();
}