#include "NetworkTrainingDialog.h"

NetworkTrainingDialog::NetworkTrainingDialog(wxWindow *parent,
                                             wxWindowID id,
                                             const std::string &title,
                                             const wxPoint &pos,
                                             const wxSize &size) : wxDialog(parent, id, title, pos, size)
{
    auto main_sizer = new wxBoxSizer(wxVERTICAL);

    plot_soluzione = new ChartControl(
        this,
        wxID_ANY,
        wxDefaultPosition,
        wxDefaultSize,
        ChartType::training);

    plot_soluzione->title = "Neural Network";

    plot_loss_function = new ChartControl(
        this,
        wxID_ANY,
        wxDefaultPosition,
        wxDefaultSize,
        ChartType::loss_function);

    plot_loss_function->title = "Loss Function";

    main_sizer->Add(plot_soluzione, 2, wxEXPAND | wxALL, FromDIP(5));
    main_sizer->Add(plot_loss_function, 1, wxEXPAND | wxALL, FromDIP(5));

    SetSizer(main_sizer);
    SetMinClientSize(size);

    Bind(wxEVT_CLOSE_WINDOW, &NetworkTrainingDialog::OnClose, this);
}

void NetworkTrainingDialog::StartTraining(NeuralNetwork *Network)
{
    if (!isProcessing)
    {
        isProcessing = true;

        auto training_function = [this](NeuralNetwork *Network)
        {
            Network->Train(
                OptimizationAlgorithm::FullBatchGradientDescent,
                plot_soluzione,
                plot_loss_function,
                QuitRequest);

            this->CallAfter([this]()
                            {
                        background_thread.join();
                        isProcessing = false;
                        EndModal(wxID_OK); });
        };

        background_thread = std::thread(training_function, Network);
    }
}

NetworkTrainingDialog::~NetworkTrainingDialog()
{
    QuitRequest = true;
    if (background_thread.joinable())
        background_thread.join();
}

void NetworkTrainingDialog::OnClose(wxCloseEvent &event)
{
#ifdef _DEBUG
    std::cout << "exiting from the application\n";
#endif

    if (isProcessing)
    {
        event.Veto();
        QuitRequest = true;
    }
    /*
    else
    {
        EndModal(wxID_CANCEL);
    }*/
}
