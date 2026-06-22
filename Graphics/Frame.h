#pragma once
#include <atomic>
#include <thread>
#include <wx/wx.h>
#include "NetworkTrainingDialog.h"

wxDECLARE_EVENT(EVT_PARAMETRI_VALIDI, wxCommandEvent);

class MyFrame : public wxFrame
{
public:
    MyFrame(const wxString& title, const wxPoint& pos, const wxSize& size);
private:
    void StartTraining(wxCommandEvent&);
    void CreateWindow();
	void SetUpReteNeurale(wxCommandEvent& event);
    void OnClose(wxCloseEvent& event);
    void AggiornaParametri(wxCommandEvent& event);
    void CheckParametri(wxCommandEvent& event);
    void OnSceltaFdAStratiNascosti(wxCommandEvent& e);
    void OnSceltaFdAOutput(wxCommandEvent& e);

#ifdef _DEBUG
    void PrintWindowsSize(wxSizeEvent& event);
#endif
    virtual ~MyFrame();

private:
    // ---- stato / logica ----
    //ChartControl* plot_soluzione        = nullptr;
    //ChartControl* plot_loss_function    = nullptr;
    //bool isProcessing                   = false;
    //std::atomic_bool QuitRequest        = false;
    NeuralNetwork* Network              = nullptr;
    //std::thread background_thread;

    // ---- Architettura ----
    wxTextCtrl* m_text_ctrl_numero_input                = nullptr;
    wxTextCtrl* m_text_ctrl_strati_nascosti             = nullptr;
    wxTextCtrl* m_text_ctrl_neuroni_per_strato_nascosto = nullptr;
    wxChoice* m_choice_strati_nascosti                  = nullptr;
    wxTextCtrl* m_text_ctrl_output                      = nullptr;
    wxChoice* m_choice_output_activation                = nullptr;
    wxString m_numero_input;
    wxString m_numero_strati_nascosti;
    wxString m_numero_neuroni_per_strato_nascosto;
    wxString m_numero_output;
    TipoDiFunzione m_tipo_funzione_strati_nascosti;
    TipoDiFunzione m_tipo_funzione_output;

    // ---- Riassunto rete ----
    wxStaticText* m_static_text_numero_pesi         = nullptr;
    wxStaticText* m_static_text_numero_bias         = nullptr;
    wxStaticText* m_static_text_parametri_totali    = nullptr;

    // ---- Training ----
    wxChoice* m_choice_tipo_di_metodo       = nullptr;
    wxTextCtrl* m_text_ctrl_numero_epoche   = nullptr;
    wxTextCtrl* m_text_ctrl_learning_rate   = nullptr;
    wxChoice* m_choice_parallelizzazione    = nullptr;
    wxString m_learning_rate;
    wxString m_numero_epoche;

    // ---- Bottoni ----
    wxButton* SetUpRete = nullptr;
    wxButton* AvviaRete = nullptr;
};