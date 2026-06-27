# Neural Network — C++ Feedforward Network with Live Training GUI

A from-scratch implementation of a fully-connected feedforward neural network in C++, with a wxWidgets GUI that visualises loss and predictions in real time as the network trains.

---

## Demo

**Function approximation — sine**

![Training demo - sine function](assets/sine.gif)

**Function approximation — exponential**

![Training demo - exponential function](assets/exp.gif)

**Function approximation — absolute value**

![Training demo - absolute value function](assets/abs.gif)

---

## Features

- Fully-connected feedforward network built from scratch (no external ML libraries)
- Configurable architecture: number of hidden layers and neurons per layer
- **Activation functions**: Linear, ReLU, Sigmoid, Tanh, Quadratic
- **Optimiser**: Full-batch gradient descent
- Real-time training dialog with live loss curve and prediction vs. target chart
- Pre-generated training datasets for 11 functions: linear, quadratic, cubic, sine, cosine, exponential, sigmoid, tanh, Gaussian, sinc, absolute value
- **WIP**: MNIST digit classification dataset support and parallelization with OpenMP and OpenCL

---

## Project Structure

```
NeuralNetwork/
├── CMakeLists.txt             # Build system
├── main.cpp                   # Entry point
├── NeuralNetwork/             # Core neural network
│   ├── NeuralNetwork.h/cpp    # Training loop, forward/backward pass, weight update
│   ├── Layer.h/cpp            # Layer container (input / hidden / output)
│   ├── Neuron.h/cpp           # Single neuron with bias and activation
│   └── ActivationFunction.h/cpp
├── Graphics/                  # wxWidgets GUI
│   ├── Frame.h/cpp            # Main window — network config and data selection
│   ├── NetworkTrainingDialog.h/cpp  # Live training dialog (threaded)
│   └── chartcontrol.h/cpp     # Custom chart widget (loss + prediction)
├── Utils/
│   └── VectorOperations.h/cpp # Vector math for gradient computations
├── assets/                    # Icons, manifests, and platform resources
└── training_data/             # Pre-generated CSV datasets
```

---

## Requirements

| Dependency | Version |
|---|---|
| CMake | 3.14+ |
| C++ compiler | C++17 (MSVC, GCC, Clang) |
| Git | any (used by FetchContent) |

wxWidgets is fetched and built automatically by CMake — no manual installation needed.

---

## Build

### Windows

```bash
cmake -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```

The executable is written to `build/Release/`.

### Linux

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

The executable is written to `build/`.

---

## Usage

1. Launch the application.
2. In the main window, select the desired function.
3. Set the network architecture: hidden layers, neurons per layer, activation functions.
4. Choose an optimiser and learning rate.
5. Click **Setup** and then **Run** — a dialog opens showing the loss curve and the current prediction overlaid on the target function, updated live each epoch.

---

## Training Data

Pre-generated datasets live in `training_data/`. Each `.txt` contains `(x, y)` pairs sampled from the corresponding function.

Supported functions: `linear`, `quadratic`, `cubic`, `sin`, `cos`, `exp`, `sigmoid`, `tanh`, `gaussian`, `sinc`, `abs`.
