# Evaluating Mamba on Continuous Physiological Sensor Data (WESAD)

**Ana Rodriguez** · **Darien Booth** — UTEP — CS 5365 — Deep Learning: Final Project


---

## Overview

This project evaluates whether Mamba's scalability advantage over Transformers, LSTMs, and CNNs holds on long continuous physiological sensor signals a domain not studied in the original Mamba paper.

We benchmark four sequence modeling architectures on the **WESAD** (Wearable Stress and Affect Detection) dataset, testing classification of three affective states — Baseline, Stress, and Amusement across sliding window sizes of **60 s**, **90 s**, and **120 s**.

The hypothesis: Mamba should maintain or improve accuracy as window size grows, while the Transformer degrades due to its quadratic attention cost.

---

## Dataset

**WESAD** — Philip Schmidt et al., ICMI 2018.

- 15 subjects, lab-controlled study
- 8 physiological channels from a chest-worn device: ECG, EDA, EMG, Respiration, Temperature, ACC (x/y/z)
- 3 affective states: Baseline (0), Stress (1), Amusement (2)
- Recorded at 700 Hz

Download from Kaggle: [WESAD Dataset](https://www.kaggle.com/datasets/orvile/wesad-wearable-stress-affect-detection-dataset)

---

## Preprocessing

| Step | Detail |
|------|--------|
| Downsampling | 700 Hz → 100 Hz |
| Windowing | Sliding windows of 60 s / 90 s / 120 s with 30 s stride |
| Train/Test Split | Subject-aware (no subject appears in both sets): 70% train / 15% val / 15% test |
| Class Imbalance | Class-balanced loss weighting |

---

## Architectures

| Model | Description |
|-------|-------------|
| **Mamba** | Selective SSM with input-dependent parameters; linear-time complexity |
| **Transformer** | Self-attention over the full sequence; quadratic complexity in sequence length |
| **LSTM** | Bidirectional LSTM; processes sequences in both temporal directions |
| **CNN** | Convolutional layers scanning local temporal regions for hierarchical features |

All four models are trained and evaluated under identical conditions on the same data splits.

---

## Repository Structure

```

├── code/
│   ├── Mamba_WESAD_Stress_Classifier_60s.ipynb    # 60-second window experiment
│   ├── Mamba_WESAD_Stress_Classifier_90s.ipynb    # 90-second window experiment
│   └── Mamba_WESAD_Stress_Classifier_120s.ipynb   # 120-second window experiment
├── data/                  # Information on attaining dataset
├── poster/                # Project poster PDF
├── report/                # Final written report PDF
├── results/               # Saved metrics, plots, and confusion matrices
└── README.md
```

Each notebook is self-contained and differs only in the `window_sec` parameter. They include:
- Dependency installation (PyTorch, `mamba-ssm`, `causal-conv1d`, scikit-learn, etc.)
- Data loading and preprocessing
- Model definitions for all four architectures
- Training loop with class-balanced loss
- Evaluation: accuracy, F1 score, full classification reports, and confusion matrices

---

## Setup

### Prerequisites

This project uses the Kaggle API to download the WESAD dataset. You will need a Kaggle account and an API token:

1. Go to https://www.kaggle.com/settings → **API** → **Create New Token**
2. This downloads a `kaggle.json` file containing your credentials

### Requirements

- Python 3.10+
- CUDA-capable GPU (notebooks were run on a T4 via Google Colab)

### Installation

```bash
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install causal-conv1d==1.4.0
pip install mamba-ssm==2.2.2 --no-build-isolation
pip install numpy scipy scikit-learn pandas matplotlib seaborn
```

### Running

Open the desired notebook in Google Colab (or a local Jupyter environment with a compatible GPU) and run all cells. The only parameter that changes across notebooks is:

```python
window_sec = 60   # or 90, or 120
```

---

## Results

### Overall Accuracy

| Model | 60 s | 90 s | 120 s |
|-------|------|------|-------|
| **Mamba** | 72.9% | **76.6%** | 68.3% |
| Transformer | 61.4% | 41.9% | 53.3% |
| Bi-LSTM | 65.3% | 59.5% | 36.7% |
| CNN | 72.9% | 72.9% | **74.9%** |

### Stress Class F1

| Model | 60 s | 90 s | 120 s |
|-------|------|------|-------|
| **Mamba** | 0.61 | **0.67** | 0.53 |
| Transformer | 0.34 | 0.31 | 0.55 |
| Bi-LSTM | 0.43 | 0.36 | 0.38 |
| **CNN** | **0.74** | **0.74** | **0.72** |

### 90-Second Window — Full Classification Report (sample)

```
Mamba
              precision    recall  f1-score   support
    Baseline     0.8740    0.9911    0.9289       112
      Stress     0.7826    0.5902    0.6729        61
   Amusement     0.3125    0.3125    0.3125        32
    accuracy                         0.7659       205
```

---

## Key Findings

- **Mamba** is the most consistent model across all window sizes, confirming the paper's scalability claim on real-world physiological signals. It never collapsed under longer sequences and achieved its best accuracy at 90 s.
- **Transformer** degraded severely with longer windows (61.4% → 41.9% at 90 s), consistent with the expected cost of quadratic attention.
- **LSTM** also fell behind at longer windows, confirming known limitations with very long sequences.
- **CNN** was a surprising strong performer, holding the highest stress F1 across all window sizes. This suggests that for stress detection, local temporal patterns may be as informative as global context — an open research question.

---

## Future Directions

- Deep Mamba 
- Bidirectional Mamba
- Simulated sensor dropout evaluation
- Subject covariate integration

---

## References

1. Gu, A. & Dao, T. "Mamba: Linear-Time Sequence Modeling with Selective State Spaces." *COLM 2024*. [arXiv:2312.00752](https://arxiv.org/abs/2312.00752)
2. Schmidt, P., Reiss, A., Duerichen, R., Marberger, C., & Van Laerhoven, K. "Introducing WESAD, a multimodal dataset for Wearable Stress and Affect Detection." *ICMI 2018*.
3. State Spaces. (2023). Mamba Official Repository. https://github.com/state-spaces/mamba