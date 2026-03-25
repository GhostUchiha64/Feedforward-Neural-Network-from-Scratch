# Aircraft Pitch Prediction: Feedforward Neural Network from Scratch

**DATA 527: Predictive Modeling — Team Project 2 (Spring 2025)** | Sri Vamsi Chitta, Prabhjyot Chavan, Siddartha Bandi

---

## Overview

This project implements a **2-hidden-layer Feedforward Neural Network (FFNN)** entirely from scratch using NumPy to predict **aircraft pitch** from real flight telemetry data. Forward propagation, backpropagation, and weight updates are all hand-coded — without using any deep learning framework (no TensorFlow or PyTorch).

The project compares three activation functions (**Sigmoid, Tanh, ReLU**) and two training modes (**Batch Gradient Descent** vs. **Stochastic Gradient Descent**), with early stopping to prevent overfitting.

---

## Dataset

| File | Description |
|------|-------------|
| `input_fl_12477` | Raw flight telemetry data (space-separated) |

**Selected Features:**
| Feature | Description |
|---------|-------------|
| `altitude` | Aircraft altitude |
| `indicated_airspeed` | Indicated airspeed |
| `pitch` | Current pitch angle |
| `roll` | Current roll angle |

**Target:** `next_pitch` — the pitch value at the next timestep (created via 1-step shift)

**Preprocessing:**
- `StandardScaler` (Z-score normalization) applied to all features and target
- 60/40 train-test split (`random_state=42`)

---

## Model Architecture

```
Input (4) → Hidden Layer 1 (16) → Hidden Layer 2 (8) → Output (1)
```

| Component | Detail |
|-----------|--------|
| Input size | 4 features |
| Hidden Layer 1 | 16 neurons |
| Hidden Layer 2 | 8 neurons |
| Output | 1 (next pitch) |
| Activations tested | Sigmoid, Tanh, ReLU |
| Learning rate | 0.001 |
| Loss function | Mean Squared Error (MSE) |
| Bias | Configurable (default: enabled) |

### Backpropagation (Hand-Coded)
Full chain-rule gradient computation through all three weight matrices:
```
dZ3 → dW3, db3 → dZ2 → dW2, db2 → dZ1 → dW1, db1
```

### Training Modes
- **Batch Gradient Descent** — computes gradients over full dataset per iteration
- **Stochastic Gradient Descent (SGD)** — updates weights per individual sample
- **Early Stopping** — halts training if MSE does not improve for 10 consecutive iterations

---

## Experiments & Comparison

The model was trained under all activation × training-mode combinations:

| Activation | Training Mode | Notes |
|-----------|---------------|-------|
| Sigmoid | Batch | Smooth convergence, slower |
| Sigmoid | Stochastic | Noisy but faster early learning |
| Tanh | Batch | Better gradient flow than Sigmoid |
| Tanh | Stochastic | Best overall stability observed |
| ReLU | Batch | Risk of dying neurons in some configs |
| ReLU | Stochastic | Fastest per-iteration compute |

---

## Repository Contents

```
Team Project 2/
├── README.md
├── nnmodelflight.py                                              # Main FFNN implementation
├── input_fl_12477                                               # Raw flight telemetry data
├── Extra.txt                                                    # Alternate feature config notes
├── Neural_Network_Pitch_Prediction_Report.docx                 # Final report (Word)
├── project2-SriVamsiChitta,PrabhjyotChavan,SiddarthaBandi.pdf  # Team submission (PDF)
├── project2-SriVamsiChitta,PrabhjyotChavan,SiddarthaBandi.docx # Team submission (Word)
└── Project 2.pdf                                               # Project specification
```

> **Note:** The `.zip` archive is excluded from this repository.

---

## Setup & Usage

### Install Dependencies
```bash
pip install pandas numpy matplotlib scikit-learn
```

### Run the Model
```bash
python nnmodelflight.py
```

The script will:
1. Load and preprocess `input_fl_12477`
2. Train the FFNN with each activation function (Sigmoid, Tanh, ReLU)
3. Log training MSE per iteration to `training.log`
4. Save final model weights and metrics to a parameter file
5. Plot MSE convergence curves and actual vs. predicted pitch values

---

## Key Results

- All three activation functions successfully learned pitch prediction from telemetry features
- **Tanh** achieved the best R² on the test set with smooth convergence
- Early stopping effectively prevented overfitting across all configurations
- The custom backpropagation implementation matched expected gradient behavior

---

## Technologies Used

| Category | Tools |
|----------|-------|
| Language | Python 3.x |
| Numerical Computing | NumPy |
| Data Processing | Pandas, Scikit-learn (preprocessing only) |
| Visualization | Matplotlib |
| Environment | Google Colab |

---

## Authors

**Sri Vamsi Chitta, Prabhjyot Chavan, Siddartha Bandi** (Team 10)
DATA 527: Predictive Modeling — Project 2 | Spring 2025
