#pragma once

#include <atomic>
#include <thread>

#include "../NeuralNetwork/NeuralNetwork.h"

class MyFrame : public wxFrame
{
public:
    MyFrame(const wxString& title, const wxPoint& pos, const wxSize& size);

private:
    void StartTraining();
    void SetUpNeuralNetwork();
    void OnClose(wxCloseEvent& event);

#ifdef _DEBUG
    void PrintWindowsSize(wxSizeEvent& event);
#endif

    virtual ~MyFrame();

private:
    ChartControl* plot_soluzione = nullptr;
    ChartControl* plot_loss_function = nullptr;

    bool isProcessing = false;

    std::atomic_bool QuitRequest = false;

    NeuralNetwork* Network = nullptr;

    std::thread background_thread;
};