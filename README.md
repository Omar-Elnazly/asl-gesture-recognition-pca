# 🤚 Hand Gesture Recognition Using PCA

> American Sign Language (ASL) alphabet recognition built entirely from scratch — no sklearn PCA, no sklearn KNN, no OpenCV filters. Pure NumPy and PIL.

**Digital Image Processing Course Project · Spring 2026**

---

## Overview

This project implements a complete image processing and machine learning pipeline for recognizing 29 ASL hand gesture classes (letters A–Z + `del`, `nothing`, `space`) using **Principal Component Analysis** for feature extraction and **K-Nearest Neighbors** for classification — both implemented from scratch using NumPy only.

---

## Pipeline

```
Raw Images (200×200 RGB)
        │
        ▼
  Grayscale Conversion          ← PIL on load
        │
        ▼
  Histogram Equalization        ← Manual LUT (no built-ins)
        │
        ▼
  Gaussian Noise Removal        ← Manual separable convolution (5×5, σ=1.2)
        │
        ▼
  Contrast Stretching           ← Percentile-based (p2–p98)
        │
        ▼
  Butterworth Lowpass Filter    ← Frequency domain (D0=30, order=2)
        │
        ▼
  Normalization [0, 1]
        │
        ▼
  Manual PCA (50 components)    ← Surrogate covariance trick, NumPy only
        │
        ▼
  Manual KNN Classifier         ← Euclidean distance + majority vote, NumPy only
        │
        ▼
  Evaluation & Results
```

---

## Results

| Metric | Value |
|---|---|
| Dataset | ASL Alphabet (Kaggle) |
| Classes | 29 (A–Z + del, nothing, space) |
| Samples | 2,900 (100 per class) |
| Train / Test split | 80% / 20% (stratified) |
| PCA components | 50 |
| Variance retained | 91.6% |
| Dimensionality reduction | 40,000 → 50 (800× compression) |
| KNN accuracy (K=5) | **45.17%** |
| Best KNN accuracy (K=1) | **65.69%** |

### Per-Class Highlights

| Best classes | Accuracy | Worst classes | Accuracy |
|---|---|---|---|
| nothing | 90% | W | 10% |
| F | 65% | O | 15% |
| L | 70% | T, U | 20% |
| G, P | 65% | V | 25% |

---

## Implementation Details

### Spatial Preprocessing (Phase I)

**Histogram Equalization** — Implemented from scratch with a manual lookup table:
1. Count pixel frequency histogram (256 bins)
2. Compute cumulative distribution function (CDF)
3. Normalize CDF into remapping LUT
4. Remap every pixel through the LUT

**Gaussian Filter** — Manual separable convolution (no `scipy`/`cv2`):
- 2D Gaussian factored into two 1D passes: `G(x,y) = G(x)·G(y)`
- Reduces complexity from O(H·W·k²) to O(H·W·2k)
- Fully vectorized using NumPy stride tricks

**Median Filter** — Manual nested-loop implementation demonstrated on a 64×64 crop.

**Contrast Stretching** — Percentile clipping at p2–p98, mapped to [0, 255].

**Butterworth Lowpass Filter** — Frequency domain filtering:
```
H(u,v) = 1 / (1 + (D(u,v) / D0)^(2n))
```
Applied via `np.fft.fft2` → shift → mask → `np.fft.ifft2`.

### PCA from Scratch (Feature Extraction)

Full 10-step implementation with educational walkthrough:

1. Flatten images → data matrix X of shape (N, 40000)
2. Compute mean image μ
3. Center data: Xc = X − μ
4. **Surrogate covariance trick** — compute (N×N) matrix L = Xc·XcᵀT instead of the infeasible (40000×40000) true covariance (saves ~12 GB RAM)
5. Eigendecomposition: `np.linalg.eigh(L)`
6. Sort eigenvalues descending
7. Project back to pixel space: u_i = Xcᵀ · v_i (normalize)
8. Compute explained variance ratio: EVR_i = λ_i / Σλ
9. Project data: z = (x − μ) · Uᵏᵀ
10. Reconstruct: x̂ = z · Uᵏ + μ

### KNN from Scratch (Classification)

No `sklearn.KNeighborsClassifier` used anywhere.

```python
# Vectorized Euclidean distance (no loops over training set)
diff = X_train - x_test_single       # (N_train, D) broadcast
dists = np.sqrt((diff ** 2).sum(axis=1))

# K nearest neighbors → majority vote
knn_idx = np.argsort(dists)[:k]
counts  = np.bincount(y_train[knn_idx])
label   = np.argmax(counts)
```

---

## Output Files

All figures are saved to the `outputs/` folder (created automatically):

| File | Description |
|---|---|
| `dataset_samples.png` | One sample image per class |
| `phase1_histogram_equalization.png` | Before/after HEQ + histograms |
| `phase1_gaussian_filter.png` | Before/after Gaussian smoothing |
| `median_filter_comparison.png` | Manual crop vs scipy full-image |
| `phase1_contrast_stretching.png` | Before/after contrast stretch |
| `phase1_frequency_domain.png` | FFT magnitude, phase, mask, output |
| `phase1_cutoff_comparison.png` | Butterworth D0 = 10, 30, 60, 100 |
| `phase1_full_spatial_pipeline_p1/2/3.png` | Full pipeline per class |
| `phase1_pca_variance.png` | Scree plot + cumulative variance |
| `phase1_eigengestures.png` | Top 10 principal components |
| `pca_mean_image.png` | Mean of all training images |
| `pca_explained_variance_stepwise.png` | Step-by-step variance plot |
| `pca_reconstruction.png` | Original vs reconstructed vs diff |
| `pca_reconstruction_k_comparison.png` | Reconstruction at k=1,5,10,25,50 |
| `class_distribution.png` | Train/test class balance |
| `sample_predictions.png` | 20 sample predictions with borders |
| `confusion_matrix.png` | Full 29×29 confusion matrix |
| `per_class_accuracy.png` | Per-class accuracy bar chart |
| `knn_k_sweep.png` | Accuracy vs K (1–20) |

---

## Requirements

```
numpy
matplotlib
Pillow
scipy
scikit-learn   ← used only for train_test_split, accuracy_score, confusion_matrix
seaborn
```

Install with:

```bash
pip install numpy matplotlib Pillow scipy scikit-learn seaborn
```

---

## Dataset

[![Kaggle](https://img.shields.io/badge/Kaggle-ASL%20Alphabet-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)

**ASL Alphabet** by [grassknoted](https://www.kaggle.com/grassknoted) on Kaggle.

| Property | Value |
|---|---|
| Source | [kaggle.com/datasets/grassknoted/asl-alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) |
| Classes | 29 — letters A–Z + `del`, `nothing`, `space` |
| Total images | ~87,000 (~3,000 per class) |
| Image size | 200×200 pixels, RGB |
| Format | JPG, organized in per-class subfolders |
| Samples used | 100 per class → 2,900 images total |

### Download

```bash
# Using Kaggle CLI
kaggle datasets download -d grassknoted/asl-alphabet
unzip asl-alphabet.zip -d data/
```

Or download manually from the link above and place it under `data/`.

### Setup

Place the unzipped dataset inside a `data/` folder in the project root so the structure looks like this:

```
data/
└── asl_alphabet_train/
    └── asl_alphabet_train/
        ├── A/
        │   ├── A1.jpg
        │   ├── A2.jpg
        │   └── ...  (~3000 images)
        ├── B/
        │   └── ...
        ├── C/
        ├── D/
        ├── E/
        ├── F/
        ├── G/
        ├── H/
        ├── I/
        ├── J/
        ├── K/
        ├── L/
        ├── M/
        ├── N/
        ├── O/
        ├── P/
        ├── Q/
        ├── R/
        ├── S/
        ├── T/
        ├── U/
        ├── V/
        ├── W/
        ├── X/
        ├── Y/
        ├── Z/
        ├── del/
        ├── nothing/
        └── space/
```

> The notebook auto-detects the dataset path across Windows, Linux, macOS, Google Colab, and Kaggle — it handles nested folder structures automatically. If auto-detection fails, it will prompt you to paste the path manually.

---

## Usage

```bash
jupyter notebook asl_main.ipynb
```

Run all cells top to bottom. The `outputs/` folder will be created automatically and all 20 figures saved there.

---

## Project Structure

```
├── asl_main.ipynb             # Main notebook — full pipeline
├── README.md
├── data/                      # Dataset (download separately — not in repo)
│   └── asl_alphabet_train/
│       └── asl_alphabet_train/
│           ├── A/             # ~3000 images per class
│           ├── B/
│           ├── ...
│           └── space/
└── outputs/                   # Auto-generated on run (not in repo)
    ├── dataset_samples.png
    ├── phase1_histogram_equalization.png
    ├── phase1_gaussian_filter.png
    ├── median_filter_comparison.png
    ├── phase1_contrast_stretching.png
    ├── phase1_frequency_domain.png
    ├── phase1_cutoff_comparison.png
    ├── phase1_full_spatial_pipeline_p1.png
    ├── phase1_full_spatial_pipeline_p2.png
    ├── phase1_full_spatial_pipeline_p3.png
    ├── phase1_pca_variance.png
    ├── phase1_eigengestures.png
    ├── pca_mean_image.png
    ├── pca_explained_variance_stepwise.png
    ├── pca_reconstruction.png
    ├── pca_reconstruction_k_comparison.png
    ├── class_distribution.png
    ├── sample_predictions.png
    ├── confusion_matrix.png
    ├── per_class_accuracy.png
    └── knn_k_sweep.png
```

---

## Key Constraints Satisfied

- **No `sklearn` PCA** — PCA implemented from scratch with surrogate covariance trick
- **No `sklearn` KNN** — KNN implemented from scratch with vectorized Euclidean distance
- **No `cv2`** — all image processing done with NumPy and PIL
- **No `histeq`** — histogram equalization implemented manually via LUT
- **Butterworth filter** — implemented via NumPy FFT, not any filter library
- **Gaussian filter** — implemented via manual separable 1D convolution
- **Median filter** — manual nested-loop implementation demonstrated

---

## Course

Digital Image Processing — Spring Term 2026
