#include "Graphics\Frame.h"

class MyApp : public wxApp
{
public:
    virtual bool OnInit() override;
};

wxIMPLEMENT_APP_CONSOLE(MyApp);

bool MyApp::OnInit()
{
    MyFrame* frame = new MyFrame("Neural network", wxPoint(0, 0), wxSize(1800, 930));
    frame->Show(true);
    return true;
}