#pragma once

#include <wx/wx.h>
#include <thread>
#include "chartcontrol.h"
#include "../NeuralNetwork/NeuralNetwork.h"

class NetworkTrainingDialog : public wxDialog
{
public:
    NetworkTrainingDialog(wxWindow *parent,
                          wxWindowID id,
                          const std::string &title,
                          const wxPoint &pos = wxDefaultPosition,
                          const wxSize &size = wxDefaultSize);
    ~NetworkTrainingDialog();
    void StartTraining(NeuralNetwork *Network);

private:
    ChartControl *plot_soluzione = nullptr;
    ChartControl *plot_loss_function = nullptr;
    bool isProcessing = false;
    std::atomic_bool QuitRequest = false;
    std::thread background_thread;

    void OnClose(wxCloseEvent &event);
};
