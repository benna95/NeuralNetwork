#include "Frame.h"

MyFrame::MyFrame(const wxString& title, const wxPoint& pos, const wxSize& size)
    : wxFrame(NULL, wxID_ANY, title, pos, size)
{
   
    SetupPannelloSuperiore();
    /*
    wxBoxSizer* button_sizer    = new wxBoxSizer(wxHORIZONTAL);

    NeuralNetworkButton = new wxButton(this, wxID_ANY, "Run", wxDefaultPosition, wxDefaultSize);
    SetUpNeuralNetworkButton = new wxButton(this, wxID_ANY, "SetUp", wxDefaultPosition, wxDefaultSize);
    NeuralNetworkButton->Bind(wxEVT_BUTTON, &MyFrame::StartTraining, this);
    SetUpNeuralNetworkButton->Bind(wxEVT_BUTTON, &MyFrame::SetUpNeuralNetwork, this);

    chart = new ChartControl(this, wxID_ANY, wxDefaultPosition, wxSize(640, 480));
    chart->title = "Neural Network";

    button_sizer->Add(NeuralNetworkButton,      0, wxALIGN_CENTER, FromDIP(5));
    button_sizer->Add(SetUpNeuralNetworkButton, 0, wxALIGN_CENTER, FromDIP(5));
    main_sizer->Add(button_sizer, 0);
    main_sizer->Add(chart, 1, wxEXPAND | wxALL, FromDIP(5));

    if (!setup)
    {
        NeuralNetworkButton->Disable();
    }

    */

    //SetSizerAndFit(main_sizer);

    
}

void MyFrame::SetupPannelloSuperiore()
{
    wxPanel* panel1 = new wxPanel(this, wxID_ANY, wxDefaultPosition, wxSize(1280, 480));

    auto mainSizer = new wxBoxSizer(wxHORIZONTAL);

    // Colonna Input
    auto inputSizer = new wxBoxSizer(wxVERTICAL);
    wxStaticText* inputTitle = new wxStaticText(panel1, wxID_ANY, "Input");

    wxSpinCtrl* inputNeuronCount = new wxSpinCtrl(
        panel1,
        wxID_ANY,
        "",
        wxDefaultPosition,
        wxDefaultSize,
        wxSP_ARROW_KEYS,
        1,
        100,
        1
    );

    inputSizer->Add(inputTitle, 0, wxALIGN_CENTER | wxBOTTOM, FromDIP(8));
    inputSizer->Add(inputNeuronCount, 0, wxALIGN_CENTER);

    // Colonna Strati nascosti
    auto hiddenSizer = new wxBoxSizer(wxVERTICAL);
    wxStaticText* hiddenTitle = new wxStaticText(panel1, wxID_ANY, "Strati nascosti");

    wxSpinCtrl* hiddenLayerCount = new wxSpinCtrl(
        panel1,
        wxID_ANY,
        "",
        wxDefaultPosition,
        wxDefaultSize,
        wxSP_ARROW_KEYS,
        1,
        20,
        3
    );

    hiddenSizer->Add(hiddenTitle, 0, wxALIGN_CENTER | wxBOTTOM, FromDIP(8));
    hiddenSizer->Add(hiddenLayerCount, 0, wxALIGN_CENTER);

    // Colonna Output
    auto outputSizer = new wxBoxSizer(wxVERTICAL);
    wxStaticText* outputTitle = new wxStaticText(panel1, wxID_ANY, "Output");

    wxSpinCtrl* outputNeuronCount = new wxSpinCtrl(
        panel1,
        wxID_ANY,
        "",
        wxDefaultPosition,
        wxDefaultSize,
        wxSP_ARROW_KEYS,
        1,
        100,
        1
    );

    outputSizer->Add(outputTitle, 0, wxALIGN_CENTER | wxBOTTOM, FromDIP(8));
    outputSizer->Add(outputNeuronCount, 0, wxALIGN_CENTER);

    // Aggiungo le colonne al sizer principale
    mainSizer->Add(inputSizer, 1, wxEXPAND | wxALL, FromDIP(10));
    mainSizer->Add(hiddenSizer, 1, wxEXPAND | wxALL, FromDIP(10));
    mainSizer->Add(outputSizer, 1, wxEXPAND | wxALL, FromDIP(10));

    panel1->SetSizerAndFit(mainSizer);
}

void MyFrame::StartTraining(wxCommandEvent&)
{
    auto training_function = [this]()
        {
            Network->Train(MetodoDiAddestramento::FullBatchGradientDescent, chart);

            this->CallAfter([this]()
                {
                    wxMessageBox("Training completato");

                    if (background_thread.joinable())
                        background_thread.join();
                });
        };

    background_thread = std::thread(training_function);    
    //N->Train(MetodoDiAddestramento::FullBatchGradientDescent, chart);
}

void MyFrame::SetUpNeuralNetwork(wxCommandEvent& event)
{
    Network = new NeuralNetwork(1, 1000, 5e-3f, 2000);

    Network->LeggiDati("generate_data/data.txt");

    Network->Add({ TipoDiStrato::input,		1, TipoDiFunzione::Lineare });

    for (int i = 0; i < 2; i++)
    {
        Network->Add({ TipoDiStrato::nascosto,	1, TipoDiFunzione::TangenteIperbolica });
    }

    Network->Add({ TipoDiStrato::output,	1, TipoDiFunzione::Lineare });

    Network->InizializzaPesieBias();
    Network->CreaMatriceConnessioni();

    //N->EsportaDati();

    setup = true;
    NeuralNetworkButton->Enable();

}