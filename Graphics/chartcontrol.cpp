#include "chartcontrol.h"

#include <wx/dcbuffer.h>
#include <wx/graphics.h>
#include <wx/settings.h>

// wxFULL_REPAINT_ON_RESIZE needed for Windows
ChartControl::ChartControl(
    wxWindow* parent,
    wxWindowID id,
    const wxPoint& pos,
    const wxSize& size,
    TipodiGrafico tipo
)
    : wxWindow(parent, id, pos, size, wxFULL_REPAINT_ON_RESIZE),
    numero_epoca(0),
    grafico(tipo)
{
    SetBackgroundStyle(wxBG_STYLE_PAINT);

    Bind(wxEVT_PAINT, &ChartControl::OnPaint, this);

    if (grafico == TipodiGrafico::soluzione)
    {
        // y_pred e y_target
        values = std::vector<std::vector<float>>(2);
    }
    else
    {
        // numero_iterazioni, loss_function
        values = std::vector<std::vector<float>>(1);
    }
}

void ChartControl::OnPaint(wxPaintEvent& evt)
{
#ifdef _DEBUG
    std::cout << "OnPaint called\n";
#endif // _DEBUG

    wxAutoBufferedPaintDC dc(this);
    dc.Clear();

    dc.SetFont(*wxNORMAL_FONT);
    const double normalFontH = static_cast<double>(dc.GetCharHeight());

    if (grafico == TipodiGrafico::soluzione)
    {
        if (values[1].size() <= 0)
            return;
    }
    else
    {
        if (values[0].size() <= 0)
            return;
    }

    wxGraphicsContext* gc = wxGraphicsContext::Create(dc);

    if (gc)
    {
        /*--------------------- font ---------------------*/

        wxFont titleFont = wxFont(
            wxNORMAL_FONT->GetPointSize() * 2.0,
            wxFONTFAMILY_DEFAULT,
            wxFONTSTYLE_NORMAL,
            wxFONTWEIGHT_BOLD
        );

        gc->SetFont(
            titleFont,
            wxSystemSettings::GetAppearance().IsDark() ? *wxWHITE : *wxBLACK
        );

        /*--------------------- font ---------------------*/

        double tw, th;
        gc->GetTextExtent(this->title, &tw, &th);

        const double titleTopBottomMinimumMargin = this->FromDIP(10);

        // Il rettangolo è definito tramite coordinata in alto a SX,
        // larghezza e altezza.
        wxRect2DDouble fullArea{
            0,
            0,
            static_cast<double>(GetSize().GetWidth()),
            static_cast<double>(GetSize().GetHeight())
        };

        const double marginX = fullArea.GetSize().GetWidth() / 16.0;
        const double marginTop = std::max(
            fullArea.GetSize().GetHeight() / 16.0,
            titleTopBottomMinimumMargin * 2.0 + th
        );
        double labelsToChartAreaMargin = this->FromDIP(10);
        const double marginBottom = labelsToChartAreaMargin * 3.0 + normalFontH * 2.0;

        wxRect2DDouble chartArea = fullArea;

        // Partendo da fullArea, creo chartArea, cio� l'area del grafico
        // delimitata dai margini.
        chartArea.Inset(marginX, marginTop, marginX, marginBottom);

        gc->DrawText(
            this->title,
            (fullArea.GetSize().GetWidth() - tw) / 2.0,
            (marginTop - th) / 2.0
        );

        wxAffineMatrix2D normalizedToChartArea{};

        auto dx = chartArea.GetLeft();
        auto dy = chartArea.GetTop();
        auto width = chartArea.GetWidth();
        auto height = chartArea.GetHeight();

        normalizedToChartArea.Translate(dx, dy);
        normalizedToChartArea.Scale(width, height);

        // Definisco una trasformazione lineare:
        // x' = x * width  + dx
        // y' = y * height + dy

        // Valori massimi e minimi del grafico.
        double ymin = *std::min_element(values[0].begin(), values[0].end());
        double ymax = *std::max_element(values[0].begin(), values[0].end());

        double xmin = *std::min_element(x_values.begin(), x_values.end());
        double xmax = *std::max_element(x_values.begin(), x_values.end());

        const auto& [YsegmentCount, YrangeLow, YrangeHigh] =
            calculateChartSegmentCountAndRange(ymin, ymax);

        const auto& [XsegmentCount, XrangeLow, XrangeHigh] =
            calculateChartSegmentCountAndRange(xmin, xmax);

        double yValueSpan = YrangeHigh - YrangeLow;
        double xValueSpan = XrangeHigh - XrangeLow;

        ymin = YrangeLow;
        ymax = YrangeHigh;

        xmin = XrangeLow;
        xmax = XrangeHigh;

        // Quante linee disegnare.
        double yLinesCount = YsegmentCount + 1;
        double xLinesCount = XsegmentCount + 1;

        wxAffineMatrix2D normalizedToYValue{};
        normalizedToYValue.Translate(0, ymax);
        normalizedToYValue.Scale(1, -1);
        normalizedToYValue.Scale(
            static_cast<double>(values[0].size() - 1),
            yValueSpan
        );

        // y' = (y + highValue) * (-yValueSpan)
        // x' = (x + 0) * (y_target.size() - 1)

        wxAffineMatrix2D normalizedToXValue{};
        normalizedToXValue.Translate(xmin, 0);
        normalizedToXValue.Scale(xmax - xmin, 1);

        gc->SetPen(wxPen(wxColor(128, 128, 128)));

        gc->SetFont(
            *wxNORMAL_FONT,
            wxSystemSettings::GetAppearance().IsDark() ? *wxWHITE : *wxBLACK
        );

        // Asse y.
        for (int i = 0; i < yLinesCount; i++)
        {
            double normalizedLineY = static_cast<double>(i) / (yLinesCount - 1);

            auto lineStartPoint = normalizedToChartArea.TransformPoint({ 0, normalizedLineY });
            auto lineEndPoint = normalizedToChartArea.TransformPoint({ 1, normalizedLineY });

            wxPoint2DDouble linePoints[] = {
                lineStartPoint,
                lineEndPoint
            };

            gc->StrokeLines(2, linePoints);

            double valueAtLineY =
                normalizedToYValue.TransformPoint({ 0, normalizedLineY }).m_y;

            auto text = wxString::Format("%.2f", valueAtLineY);

            text = wxControl::Ellipsize(
                text,
                dc,
                wxELLIPSIZE_MIDDLE,
                chartArea.GetLeft() - labelsToChartAreaMargin
            );

            double tw, th;
            gc->GetTextExtent(text, &tw, &th);

            gc->DrawText(
                text,
                chartArea.GetLeft() - labelsToChartAreaMargin - tw,
                lineStartPoint.m_y - th / 2.0
            );
        }

        // Asse x.
        for (int i = 0; i < xLinesCount; i++)
        {
            double normalizedLineX = static_cast<double>(i) / (xLinesCount - 1);

            auto lineStartPoint = normalizedToChartArea.TransformPoint({ normalizedLineX, 0 });
            auto lineEndPoint = normalizedToChartArea.TransformPoint({ normalizedLineX, 1 });

            wxPoint2DDouble linePoints[] = {
                lineStartPoint,
                lineEndPoint
            };

            gc->StrokeLines(2, linePoints);

            double valueAtLineX =
                normalizedToXValue.TransformPoint({ normalizedLineX, 0 }).m_x;

            wxString text;

            if (grafico == TipodiGrafico::soluzione)
            {
                text = wxString::Format("%.2f", valueAtLineX);
            }
            else
            {
                text = wxString::Format("%.0f", valueAtLineX);
            }

            text = wxControl::Ellipsize(
                text,
                dc,
                wxELLIPSIZE_MIDDLE,
                chartArea.GetLeft() - labelsToChartAreaMargin
            );

            double tw, th;
            gc->GetTextExtent(text, &tw, &th);

            gc->DrawText(
                text,
                lineStartPoint.m_x - tw / 2,
                chartArea.GetBottom() + labelsToChartAreaMargin
            );
        }

        /*----------------- numero epoca corrente ----------------*/

        wxFont legendFont = wxFont(
            wxNORMAL_FONT->GetPointSize(),
            wxFONTFAMILY_DEFAULT,
            wxFONTSTYLE_NORMAL,
            wxFONTWEIGHT_BOLD
        );

        gc->SetFont(
            legendFont,
            wxSystemSettings::GetAppearance().IsDark() ? *wxWHITE : *wxBLACK
        );

        auto testo_iterazione = wxString::Format("Epoch: %d", numero_epoca);

        gc->GetTextExtent(testo_iterazione, &tw, &th);

        auto testo_iterazione_xcoord = 1.05 * chartArea.GetLeft();
        auto testo_iterazione_ycoord = 1.05 * chartArea.GetTop();

        gc->DrawText(testo_iterazione, testo_iterazione_xcoord, testo_iterazione_ycoord);

        /*----------------- numero epoca corrente ----------------*/

        gc->SetPen(wxPen(wxColor(128, 128, 128)));

        gc->SetFont(
            *wxNORMAL_FONT,
            wxSystemSettings::GetAppearance().IsDark() ? *wxWHITE : *wxBLACK
        );

        /*------------------------ legenda -----------------------*/

        if (grafico == TipodiGrafico::soluzione)
        {
            const double lineLen      = this->FromDIP(25);
            const double symbolRadius = this->FromDIP(5);
            const double gap          = this->FromDIP(5);
            const double itemSpacing  = this->FromDIP(10);

            double twT, thT, twP, thP;
            gc->SetFont(*wxNORMAL_FONT,
                wxSystemSettings::GetAppearance().IsDark() ? *wxWHITE : *wxBLACK);
            gc->GetTextExtent("Target",     &twT, &thT);
            gc->GetTextExtent("Prediction", &twP, &thP);

            const double legendY = chartArea.GetBottom() + labelsToChartAreaMargin * 2.0 + normalFontH;
            const double totalWidth  = lineLen + gap + twT + itemSpacing + lineLen + gap + twP;
            double x = (fullArea.GetWidth() - totalWidth) / 2.0;

            // Target: puntini blu
            gc->SetPen(*wxTRANSPARENT_PEN);
            gc->SetBrush(*wxBLUE_BRUSH);
            gc->DrawEllipse(x + lineLen / 2.0 - symbolRadius,
                            legendY - symbolRadius,
                            symbolRadius * 2, symbolRadius * 2);
            gc->DrawText("Target", x + lineLen + gap, legendY - thT / 2.0);
            x += lineLen + gap + twT + itemSpacing;

            // Predizione: linea rossa
            gc->SetPen(wxPen(*wxRED, 3));
            gc->StrokeLine(x, legendY, x + lineLen, legendY);
            gc->DrawText("Prediction", x + lineLen + gap, legendY - thP / 2.0);
        }

        /*------------------------ legenda -----------------------*/

        gc->SetPen(wxPen(wxColor(128, 128, 128)));

        gc->SetFont(
            *wxNORMAL_FONT,
            wxSystemSettings::GetAppearance().IsDark() ? *wxWHITE : *wxBLACK
        );

        wxPoint2DDouble leftHLinePoints[] = {
            normalizedToChartArea.TransformPoint({ 0, 0 }),
            normalizedToChartArea.TransformPoint({ 0, 1 })
        };

        wxPoint2DDouble rightHLinePoints[] = {
            normalizedToChartArea.TransformPoint({ 1, 0 }),
            normalizedToChartArea.TransformPoint({ 1, 1 })
        };

        gc->StrokeLines(2, leftHLinePoints);
        gc->StrokeLines(2, rightHLinePoints);

        wxPoint2DDouble* pointArray = nullptr;
        wxPoint2DDouble* pointArray2 = nullptr;

        if (grafico == TipodiGrafico::soluzione)
        {
            pointArray = new wxPoint2DDouble[values[0].size()];
            pointArray2 = new wxPoint2DDouble[values[1].size()];
        }
        else
        {
            pointArray = new wxPoint2DDouble[values[0].size()];
        }

        wxAffineMatrix2D valueToNormalized = normalizedToYValue;
        valueToNormalized.Invert();

        wxAffineMatrix2D valueToChartArea = normalizedToChartArea;
        valueToChartArea.Concat(valueToNormalized);

        for (int i = 0; i < values[0].size(); i++)
        {
            pointArray[i] = valueToChartArea.TransformPoint({
                static_cast<double>(i),
                values[0][i]
                });

            if (grafico == TipodiGrafico::soluzione)
            {
                pointArray2[i] = valueToChartArea.TransformPoint({
                    static_cast<double>(i),
                    values[1][i]
                    });
            }
        }

        gc->SetBrush(wxBrush(
            wxSystemSettings::GetAppearance().IsDark() ? *wxCYAN : *wxBLUE
        ));

        gc->SetPen(*wxTRANSPARENT_PEN);

        if (grafico == TipodiGrafico::soluzione)
        {
            for (size_t i = 0; i < values[0].size(); i++)
            {
                if (i % 5 == 0)
                {
                    gc->DrawEllipse(
                        pointArray[i].m_x - 5,
                        pointArray[i].m_y - 5,
                        10,
                        10
                    );
                }
            }

            gc->SetPen(wxPen(
                wxSystemSettings::GetAppearance().IsDark() ? *wxRED : *wxRED,
                3
            ));

            gc->StrokeLines(values[1].size(), pointArray2);
        }
        else
        {
            gc->SetPen(wxPen(
                wxSystemSettings::GetAppearance().IsDark() ? *wxRED : *wxRED,
                3
            ));

            gc->StrokeLines(numero_epoca, pointArray);
        }

        delete[] pointArray;

        if (grafico == TipodiGrafico::soluzione)
        {
            delete[] pointArray2;
        }

        delete gc;
    }
}

std::tuple<int, double, double> ChartControl::calculateChartSegmentCountAndRange(
    double origLow,
    double origHigh
)
{
    constexpr double rangeMults[] = {
        0.2,
        0.25,
        0.5,
        1.0,
        2.0,
        2.5,
        5.0
    };

    constexpr int maxSegments = 6;

    double magnitude = std::floor(std::log10(origHigh - origLow));

    for (auto r : rangeMults)
    {
        double stepSize = r * std::pow(10.0, magnitude);

        double low = std::floor(origLow / stepSize) * stepSize;
        double high = std::ceil(origHigh / stepSize) * stepSize;

        int segments = round((high - low) / stepSize);

        if (segments <= maxSegments)
        {
            return std::make_tuple(segments, low, high);
        }
    }

    // Return some defaults in case rangeMults and maxSegments are mismatched.
    return std::make_tuple(10, origLow, origHigh);
}