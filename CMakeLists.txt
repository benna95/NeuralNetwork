cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(NeuralNetwork LANGUAGES CXX)

include(FetchContent)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(wxBUILD_SHARED OFF)

message(STATUS "Fetching wxWidgets...")

FetchContent_Declare(
   wxWidgets
   GIT_REPOSITORY https://github.com/wxWidgets/wxWidgets.git
   GIT_SHALLOW ON
)
FetchContent_MakeAvailable(wxWidgets)

set(SRCS main.cpp
	Graphics/chartcontrol.cpp
	Graphics/Frame.cpp
	Graphics/NetworkTrainingDialog.cpp
	NeuralNetwork/ActivationFunction.cpp
	NeuralNetwork/Layer.cpp
	NeuralNetwork/NeuralNetwork.cpp
	NeuralNetwork/Neuron.cpp
	Utils/VectorOperations.cpp)
	
	
if(WIN32)
    add_executable(main WIN32 ${SRCS} assets/main.exe.manifest assets/main.rc)
else()
    add_executable(main ${SRCS})
endif()


target_link_libraries(main PRIVATE wxcore wxnet)