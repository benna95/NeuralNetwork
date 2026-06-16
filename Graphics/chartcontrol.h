#pragma once

#include <string>
#include <tuple>
#include <vector>

#include <wx/wx.h>
#include <wx/statline.h>


enum class TipodiGrafico
{
    soluzione       = 0,
    loss_function   = 1
};

class ChartControl : public wxWindow
{
public:
    ChartControl(
        wxWindow* parent,
        wxWindowID id,
        const wxPoint& pos,
        const wxSize& size,
        TipodiGrafico tipo
    );

    void OnPaint(wxPaintEvent& event);

    std::vector<float> x_values;
    std::vector<std::vector<float>> values;

    std::string title;

    int numero_epoca;

    TipodiGrafico grafico;

private:
    std::tuple<int, double, double> calculateChartSegmentCountAndRange(
        double origLow,
        double origHigh
    );
};