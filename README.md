# 🤚 Hand Gesture Recognition Using Manual PCA

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Only%20PCA%20%26%20KNN-013243?style=for-the-badge&logo=numpy&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Image%20Processing-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=for-the-badge)

**A complete hand gesture recognition pipeline built from scratch — no black-box ML shortcuts.**

_Digital Image Processing Course Project_

</div>

---

## 📖 Project Description

This project implements a **full hand gesture recognition system** for the **ASL (American Sign Language) Alphabet** dataset — entirely from the ground up, without relying on high-level ML abstractions for the core pipeline.

### What is Hand Gesture Recognition?

Hand gesture recognition is the task of automatically identifying hand shapes or movements from images. It has wide applications in sign language translation, human-computer interaction, accessibility tools, and robotics.

### Why Manual PCA?

**Principal Component Analysis (PCA)** is used to reduce the dimensionality of images before classification. Each 200×200 grayscale image contains 40,000 pixel values — far too many raw features for efficient and generalisable classification. PCA compresses this into a compact set of the most informative directions (principal components), filtering out noise and redundancy.

Crucially, **PCA is implemented fully from scratch** using only NumPy — no `sklearn.decomposition.PCA`, no `cv2.PCACompute`. This demonstrates a deep understanding of the underlying linear algebra.

Similarly, **K-Nearest Neighbors (KNN) classification is also implemented manually** from scratch — no `sklearn.KNeighborsClassifier`.

### Pipeline Overview

The system takes raw RGB images, preprocesses them through spatial and frequency domains, extracts features via manual PCA, and classifies them using manual KNN — all with evaluation and visualisation at each stage.

---

## ✨ Features

- **Spatial Domain Preprocessing** — grayscale conversion, per-pixel histogram equalization, Gaussian smoothing, median filtering, and contrast stretching, all implemented manually
- **Histogram Equalization** — manual CDF-based lookup-table remap to boost contrast uniformly across all images
- **Noise Filtering** — manual separable Gaussian convolution (stride-tricks, no loops) and manual median filter for salt-and-pepper noise
- **Frequency-Domain Filtering** — Butterworth lowpass filter applied in the FFT domain to suppress high-frequency noise while preserving edge structure
- **Manual PCA Implementation** — covariance matrix, eigendecomposition via NumPy, eigen-sorting, and projection — zero sklearn/cv2 PCA
- **Manual KNN Classification** — distance-based nearest-neighbour voting implemented entirely with NumPy
- **Evaluation Metrics** — accuracy score, per-class precision/recall/F1, confusion matrix heatmap, per-class accuracy bar chart, and K-sensitivity analysis
- **Rich Visualisation Outputs** — preprocessing comparisons, eigengestures, PCA reconstruction at multiple component counts, scatter plots, and prediction samples

---

## 🔄 Project Pipeline

```
Raw RGB Images (200×200 px)
         │
         ▼
┌─────────────────────────┐
│   Grayscale Conversion  │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Histogram Equalization │  (manual CDF-based LUT)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   Gaussian / Median     │  (manual separable convolution)
│   Noise Removal         │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   Contrast Stretching   │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Butterworth FFT Filter │  (frequency-domain lowpass)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Image Flattening       │  40,000-dim feature vectors
│  + Normalisation [0,1]  │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   Manual PCA            │  40,000 → 50 principal components
│   (NumPy only)          │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   Train / Test Split    │  (sklearn for splitting only)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   Manual KNN            │  (NumPy only, no sklearn classifier)
│   Classification        │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Evaluation & Results   │  Accuracy · Confusion Matrix · F1
└─────────────────────────┘
```

---

## 🛠️ Technologies Used

| Tool                 | Purpose                                                         |
| -------------------- | --------------------------------------------------------------- |
| **Python 3.8+**      | Core programming language                                       |
| **NumPy**            | All matrix operations, PCA, KNN — the backbone                  |
| **OpenCV (cv2)**     | Image I/O and basic utilities                                   |
| **Pillow (PIL)**     | Image loading and resizing                                      |
| **Matplotlib**       | Visualisations and figure export                                |
| **Seaborn**          | Confusion matrix heatmaps                                       |
| **SciPy**            | Median filter (for comparison only — not used in the pipeline)  |
| **Scikit-learn**     | Train/test split + evaluation metrics **only** (no PCA, no KNN) |
| **Jupyter Notebook** | Interactive development and presentation                        |

---

## 🧮 Manual PCA Implementation

> **PCA is implemented entirely from scratch using NumPy. No `sklearn.decomposition.PCA` or `cv2.PCACompute` is used anywhere in the feature extraction pipeline.**

### Steps

1. **Mean Subtraction** — compute the mean image across all training samples and subtract it to centre the data around zero
2. **Covariance Matrix** — compute `Cov = Xc.T @ Xc / (N - 1)` where `Xc` is the mean-centred data matrix
3. **Eigendecomposition** — extract eigenvalues and eigenvectors using `np.linalg.eigh` (numerically stable for symmetric matrices)
4. **Eigen-Sorting** — sort eigenvectors in descending order of their eigenvalues to rank principal components by explained variance
5. **Dimensionality Reduction** — project each sample onto the top-`k` eigenvectors: `X_pca = Xc @ components.T`
6. **Reconstruction** — optionally reconstruct images from PCA features to visualise information loss at different component counts

The result: **40,000-dimensional pixel vectors reduced to 50 principal components**, capturing the dominant variance across all 29 gesture classes.

---

## 📂 Dataset

| Property                  | Value                                                                                     |
| ------------------------- | ----------------------------------------------------------------------------------------- |
| **Source**                | [Kaggle — ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) |
| **Classes**               | 29 (letters A–Z + `del`, `nothing`, `space`)                                              |
| **Original dataset size** | ~3,000 images per class (~87,000 total)                                                   |
| **Samples used**          | 100 per class → **2,900 images** (configurable)                                           |
| **Image dimensions**      | 200×200 px, RGB → converted to grayscale                                                  |
| **Format**                | JPG, organised in per-class subfolders                                                    |

**Expected folder structure:**

```
data/
└── asl_alphabet_train/
    ├── A/    ← ~3,000 images
    ├── B/
    ├── ...
    └── space/
```

---

## 📊 Results

> Results are generated automatically when the notebook is run end-to-end. Figures are saved to the `outputs/` folder.

| Metric               | Value                                     |
| -------------------- | ----------------------------------------- |
| **Test Accuracy**    | `[Run the notebook to generate]`          |
| **PCA Components**   | 50                                        |
| **KNN — K**          | `[see hyperparameter sweep in Section 9]` |
| **Train/Test Split** | 80 / 20                                   |

### Generated Output Figures

| File                                | Description                                         |
| ----------------------------------- | --------------------------------------------------- |
| `dataset_samples.png`               | One sample image per class (29 classes)             |
| `phase1_histogram_equalization.png` | Before/after histogram equalization                 |
| `preprocessing_pipeline_pageN.png`  | Full spatial preprocessing comparison per class     |
| `frequency_domain_filter.png`       | Butterworth FFT filter visualisation                |
| `pca_explained_variance.png`        | Scree plot — variance captured per component        |
| `eigengestures.png`                 | Top principal components visualised as images       |
| `pca_reconstruction.png`            | Image reconstruction at 1, 5, 10, 20, 50 components |
| `confusion_matrix.png`              | 29×29 confusion matrix heatmap                      |
| `per_class_accuracy.png`            | Per-class accuracy bar chart                        |
| `knn_k_sweep.png`                   | Accuracy vs K neighbours sweep                      |

> 📌 _All figures are saved automatically to `outputs/` when the notebook is executed._

---

## 🗂️ Repository Structure

```
hand-gesture-recognition-pca/
│
├── asl_main.ipynb              # Main project notebook (all phases)
│
├── outputs/                    # Auto-generated figures and results
│   ├── dataset_samples.png
│   ├── phase1_histogram_equalization.png
│   ├── confusion_matrix.png
│   ├── per_class_accuracy.png
│   ├── pca_explained_variance.png
│   ├── eigengestures.png
│   ├── pca_reconstruction.png
│   └── knn_k_sweep.png
│
├── data/                       # Dataset folder (not tracked by Git)
│   └── asl_alphabet_train/
│       ├── A/
│       ├── B/
│       └── ...
│
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
└── README.md                   # This file
```

---

## ⚙️ Installation

**1. Clone the repository**

```bash
git clone <repo-url>
cd hand-gesture-recognition-pca
```

**2. (Recommended) Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Download the dataset**

Download the [ASL Alphabet dataset from Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) and place it so that the folder containing `A/`, `B/`, ..., `space/` subfolders is accessible. The notebook auto-detects the dataset path across Windows, macOS, Linux, Google Colab, and Kaggle.

---

## ▶️ Usage

**Launch Jupyter Notebook**

```bash
jupyter notebook asl_main.ipynb
```

**Or use JupyterLab**

```bash
jupyter lab asl_main.ipynb
```

Then run all cells from top to bottom (`Kernel → Restart & Run All`). The notebook is self-contained and will:

1. Auto-detect the dataset path
2. Apply the full preprocessing pipeline
3. Run manual PCA and manual KNN
4. Print evaluation metrics and save all figures to `outputs/`

> 💡 **Google Colab / Kaggle:** The notebook detects these environments automatically. Just upload it and point to the dataset.

---

## 🚀 Future Improvements

- **Real-time Recognition** — integrate webcam capture (OpenCV `VideoCapture`) for live hand gesture classification
- **CNN Baseline Comparison** — train a lightweight Convolutional Neural Network (e.g. MobileNet) and compare accuracy, inference speed, and feature quality against the PCA + KNN pipeline
- **Deep Learning Feature Extraction** — replace manual PCA with features extracted from a pretrained CNN backbone
- **Data Augmentation** — apply rotation, flipping, brightness jitter, and perspective warping to improve generalisation
- **Larger Training Set** — currently using 100 images per class; scaling to the full ~3,000 per class may significantly boost accuracy
- **Kernel PCA** — extend the manual PCA implementation to nonlinear kernels (RBF, polynomial) for improved class separability
- **Sign Language Sentence Recognition** — chain gesture predictions into word/sentence-level output using sequence models

---

## 👥 Authors

| Name            | Role                                |
| --------------- | ----------------------------------- |
| `[Author Name]` | Project development, implementation |
| `[Author Name]` | Project development, implementation |

_Digital Image Processing Course — `[University Name]`, `[Year]`_

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

This project was developed for educational purposes as part of a university Digital Image Processing course. The ASL Alphabet dataset is sourced from Kaggle and retains its original license.

---

<div align="center">

_Built with 🧠 linear algebra, 🖼️ image processing, and ☕ a lot of NumPy._

</div>
