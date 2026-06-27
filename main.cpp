#include "Graphics/Frame.h"

class MyApp : public wxApp
{
public:
    virtual bool OnInit() override;
};

#ifdef _DEBUG
wxIMPLEMENT_APP_CONSOLE(MyApp);
#else
wxIMPLEMENT_APP(MyApp);
#endif // _DEBUG

bool MyApp::OnInit()
{
    MyFrame *frame = new MyFrame("Neural network", wxDefaultPosition, wxDefaultSize);
    frame->Show(true);
    return true;
}