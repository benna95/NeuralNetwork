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
├── training_data/             # Pre-generated CSV datasets
└── assets/                    # Application icon
```

---

## Requirements

| Dependency | Version |
|---|---|
| Visual Studio | 2022 (toolset v143) |
| wxWidgets | 3.3.1 |
| C++ standard | C++17 |


---

## Build

Open `NeuralNetwork.sln` in Visual Studio 2022 and build in **Release | x64** or **Debug | x64**.

```
msbuild NeuralNetwork.sln /p:Configuration=Release /p:Platform=x64
```

The executable is written to `bin/x64/Release/`.

---

## Usage

1. Launch the application.
2. In the main window, select the desired function.
3. Set the network architecture: hidden layers, neurons per layer, activation functions.
4. Choose an optimiser and learning rate.
5. Click **Setup** and then **Run** — a dialog opens showing the loss curve and the current prediction overlaid on the target function, updated live each epoch.

---

## Training Data

Pre-generated datasets live in `training_data/`. Each .txt contains `(x, y)` pairs sampled from the corresponding function.

Supported functions: `linear`, `quadratic`, `cubic`, `sin`, `cos`, `exp`, `sigmoid`, `tanh`, `gaussian`, `sinc`, `abs`.
