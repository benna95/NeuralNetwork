#include "Frame.h"

MyFrame::MyFrame(const wxString& title, const wxPoint& pos, const wxSize& size)
    : wxFrame(NULL, wxID_ANY, title, pos, size)
{
    auto main_sizer = new wxBoxSizer(wxVERTICAL);

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

    plot_loss_function->title = "Loss Function";

    this->Bind(wxEVT_CLOSE_WINDOW, &MyFrame::OnClose, this);

#ifdef _DEBUG
    this->Bind(wxEVT_SIZE, &MyFrame::PrintWindowsSize, this);
#endif

    main_sizer->Add(plot_soluzione, 2, wxEXPAND | wxALL, FromDIP(5));
    main_sizer->Add(plot_loss_function, 1, wxEXPAND | wxALL, FromDIP(5));

    SetSizer(main_sizer);
    SetMinClientSize(size);

    SetUpNeuralNetwork();

    StartTraining();
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

void MyFrame::SetUpNeuralNetwork()
{
    Network = new NeuralNetwork(1, 1000, 1e-3f, 5000);

    Network->LeggiDati("generate_data/data.txt");

    Network->Add({ TipoDiStrato::input,    1, TipoDiFunzione::Lineare });

    for (int i = 0; i < 1; i++)
    {
        Network->Add({ TipoDiStrato::nascosto, 2, TipoDiFunzione::TangenteIperbolica });
    }

    Network->Add({ TipoDiStrato::output,   1, TipoDiFunzione::Lineare });

    Network->InizializzaPesieBias();
    Network->CreaMatriceConnessioni();
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