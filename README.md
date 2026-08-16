# CPU-Friendly Pneumonia Classification from Pediatric Chest X-Rays

This repository contains a CPU-only medical-image classification project built for the Voxelgrids AI/ML Engineer hiring assignment. A compact convolutional neural network (CNN) classifies 28 x 28 pediatric chest X-rays from PneumoniaMNIST as **Normal** or **Pneumonia**.

The goal is to demonstrate a complete, reproducible ML workflow rather than claim clinical readiness. The repository includes dataset analysis, two training experiments, evaluation, threshold analysis, failure analysis, single-image inference, and a project-aware local LLM assistant.

> **Clinical disclaimer:** This is an experimental educational project. It is not a medical device, does not provide a diagnosis, and has not been clinically or externally validated.

## Results

Both models were evaluated at a classification threshold of 0.5 on the official PneumoniaMNIST test set of 624 images.

|      Model      | Accuracy | Precision | Sensitivity | Specificity |  F1    | ROC-AUC| FP | FN |
|       ---       |    ---   |    ---    |     ---     |    ---      |  ---   |   ---  |--- |--- |
| Baseline CNN    |  89.10%  |   87.27%  |    96.67%   |   76.50%    | 91.73% | 95.95% | 55 | 13 |
| Regularized CNN |  86.38%  |   83.22%  |    97.95%   |   67.09%    | 89.99% | 96.20% | 77 |  8 |

The baseline provides the better overall balance at the default threshold, while the regularized experiment reduces false negatives at the cost of substantially more false positives. The comparison between the two test results should be interpreted as exploratory rather than as formal model selection. A stricter follow-up would select the model and threshold entirely on validation data and evaluate one locked pipeline on an untouched test set.

See [REPORT.md](REPORT.md) or `report.pdf` for the full discussion.

## Dataset

[PneumoniaMNIST](https://medmnist.com/) is a binary classification dataset derived from pediatric chest X-rays and distributed through MedMNIST.

|    Split   |  Normal   | Pneumonia |   Total   |
|     ---    |    ---    |    ---    |    ---    |
|    Train   |   1,214   |   3,494   |   4,708   |
| Validation |    135    |    389    |    524    |
|    Test    |    234    |    390    |    624    |
|  **Total** | **1,583** | **4,273** | **5,856** |

Images are grayscale and center-cropped/resized by PneumoniaMNIST to 28 x 28 pixels. The official splits are retained. The dataset is imbalanced, so accuracy is reported together with sensitivity, specificity, precision, F1, ROC-AUC, and confusion matrices.

## Model

The baseline is a lightweight custom CNN:

`1×28×28 → Conv(32) → MaxPool → Conv(64) → MaxPool → Conv(128) → AdaptiveAvgPool → Linear(128→1)`

ReLU activations are used after each convolution. The regularized
experiment uses the same feature extractor with dropout (`p=0.3`)
before the classifier and conservative training-time augmentation.

## Repository structure

```text
.
├── README.md                     # Setup, usage, and project overview
├── REPORT.md                     # Editable report source
├── report.pdf                    # Submission report
├── requirements.txt
├── dataset_analysis.py           # Dataset statistics and sample plots
├── train.py                      # Baseline training
├── train_regularized.py          # Augmentation + dropout experiment
├── evaluate.py                   # Baseline test evaluation
├── evaluate_regularized.py       # Regularized test evaluation
├── analysis.py                   # Curves, examples, and failure cases
├── thresh_analysis.py            # ROC and validation threshold analysis
├── inference.py                  # Single-image baseline inference
├── export_test_img.py            # Export one test image for demonstration
├── run_assistant.py              # CLI/web assistant launcher
├── app.py                        # Streamlit assistant UI
├── models/
│   └── cnn.py                    # Both CNN definitions
├── assistant/
│   ├── assistant.py              # Ollama integration and grounded prompt
│   └── project_context.txt       # Assistant knowledge base
├── data/
├── saved_models/
└── results/
```

## Setup

### Prerequisites

- Python 3.10 or 3.11
- A CPU is sufficient
- Ollama is required only for the AI assistant

Create a fresh environment instead of reusing an environment from another machine.

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If Python 3.11 is unavailable, replace `-3.11` with the installed compatible version.

### Linux or macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

All commands below should be run from the repository root with the environment activated.

## Reproduce the workflow

### 1. Download and inspect the dataset

```powershell
python dataset_analysis.py
```

This downloads PneumoniaMNIST when necessary and creates:

- `results/class_distribution.png`
- `results/sample_images.png`

The repository currently includes the dataset archive. To reproduce from a clean checkout without it, run this step before training.

### 2. Train the baseline

```powershell
python train.py
```

Configuration: 15 epochs, batch size 64, Adam optimizer, learning rate 0.001, `BCEWithLogitsLoss`, CPU, seed 42. The checkpoint with the lowest validation loss is written to `saved_models/best_baseline_model.pth`.

### 3. Train the regularized experiment

```powershell
python train_regularized.py
```

This experiment adds small rotations, translations, and dropout. Because augmentation and dropout are changed together, it is not an ablation study and their effects cannot be separated.

### 4. Evaluate saved checkpoints

```powershell
python evaluate.py
python evaluate_regularized.py
```

These scripts print test metrics and save confusion matrices.

### 5. Generate analysis artifacts

```powershell
python analysis.py
python thresh_analysis.py
```

These generate training curves, correct predictions, failure cases, ROC curves, and a sensitivity-specificity threshold plot. Threshold analysis uses validation data; test data is not used to tune the threshold.

## Single-image inference

Run the baseline model on the included demonstration image:

```powershell
python inference.py --image results/example_test_xray.png
```

Expected output for the included example is a pneumonia prediction with a score of approximately 0.9982.

The script resizes arbitrary inputs to 28 x 28, but this does not reproduce every aspect of PneumoniaMNIST dataset construction. Results on external X-rays should therefore be treated only as an interface demonstration. The sigmoid score is not a calibrated clinical probability.

## AI assistant

The assistant uses the local `phi3:mini` model through Ollama and is grounded with `assistant/project_context.txt`.

1. Install and start [Ollama](https://ollama.com/).
2. Download the model:

```powershell
ollama pull phi3:mini
```

3. Launch the interface selector:

```powershell
python run_assistant.py
```

Choose `0` for the command-line interface or `1` for Streamlit. The web interface can also be launched directly:

```powershell
python -m streamlit run app.py
```

The assistant answers questions about this project; it is not a clinical assistant. A local Ollama service and the downloaded model are required.

## Rebuild the report PDF

The editable report is stored in `REPORT.md`. Rebuild `report.pdf` after changing it with:

```powershell
python build_report.py
```

`report.pdf`is the submitted report. REPORT.md is its editable source.`build_report.py` is provided as an optional utility for regenerating the PDF and is not required for model training, evaluation, inference, or the AI assistant.
The builder uses PyMuPDF and appends selected plots from `results/` as figure pages.

## Main limitations

- Images are only 28 x 28, so fine radiographic detail is lost.
- The data represents pediatric patients and should not be generalized to adults.
- No external, cross-hospital, or prospective validation was performed.
- Scores were not calibrated.
- The 0.5 threshold was not chosen from a clinical operating requirement.
- The regularized experiment changed augmentation and dropout simultaneously.
- Comparing both models on the test set makes their test comparison exploratory.
- External images may differ from the dataset preprocessing and distribution.
