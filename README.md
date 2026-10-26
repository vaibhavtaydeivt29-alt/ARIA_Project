# ARIA - Adaptive Radar Intelligence Architecture

## Overview

ARIA (Adaptive Radar Intelligence Architecture) is an AI-powered radar intelligence framework designed for target classification, predictive tracking, threat assessment, and adaptive decision-making using deep learning techniques.

The system combines Convolutional Neural Networks (CNNs), Long Short-Term Memory (LSTM) networks, distributed sensor fusion, ECCM mechanisms, and a threat assessment engine into a unified radar intelligence pipeline.

---

## Features

- CNN-Based Target Classification
- LSTM-Based Predictive Tracking
- Distributed Sensor Mesh Architecture
- Electronic Counter-Counter Measures (ECCM)
- Threat Assessment Engine
- End-to-End Radar Intelligence Pipeline
- Synthetic Radar Data Generation
- Modular Architecture for Research and Experimentation

---

## Project Structure

```text
ARIA_Project/
│
├── aria/
│   ├── cnn_model.py
│   ├── lstm_model.py
│   ├── sensor_mesh.py
│   ├── eccm.py
│   ├── threat_engine.py
│   ├── pipeline.py
│   ├── preprocessing.py
│   ├── config.py
│   └── data_loader.py
│
├── data/
│   └── mstar/
│
├── models/
│
├── results/
│
├── requirements.txt
├── run_demo.py
├── train_cnn.py
├── train_lstm.py
└── README.md
```

---

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Scikit-Learn

---

## Installation

Clone the repository:

```bash
git clone https://github.com/vaibhavtaydeivt29-alt/ARIA_Project.git
```

Move into the project folder:

```bash
cd ARIA_Project
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the virtual environment (Windows):

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Training CNN Model

```bash
python train_cnn.py
```

---

## Training LSTM Model

```bash
python train_lstm.py
```

---

## Running Complete Pipeline

```bash
python run_demo.py
```

---

## Output

Generated outputs are stored in:

```text
models/
results/
```

Example outputs include:

- Trained CNN Model
- Trained LSTM Model
- CNN Loss Curves
- Confusion Matrix
- LSTM Prediction Graphs

---

## Research Modules

| Module | Description |
|----------|------------|
| CNN Classification | Radar Target Recognition |
| LSTM Tracking | Predictive Radar Tracking |
| Sensor Mesh | Distributed Sensor Fusion |
| ECCM Layer | Countermeasure Resistance |
| Threat Engine | Threat Prioritization |
| Pipeline | End-to-End Processing |

---

## Future Improvements

- Real Radar Dataset Integration
- Real-Time Streaming Support
- Advanced Target Tracking Algorithms
- Multi-Sensor Data Fusion
- Edge Deployment Support

---

## Author

**Vaibhav Tayde**

Artificial Intelligence & Machine Learning Research Project

---

## License

This project is intended for educational and research purposes.
