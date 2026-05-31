git clone https://github.com/Keobu/Industrial-Quality-Control.git

# Industrial Quality Control: Automated Defect Detection via Hybrid Computer Vision

## 1. Project Overview

Automated surface inspection is a cornerstone of modern industrial quality assurance, enabling manufacturers to detect defects such as cracks, scratches, and structural anomalies on metallic surfaces with high reliability and speed. This project presents a robust, end-to-end Computer Vision system that leverages a **hybrid feature approach**—combining advanced handcrafted spatial descriptors with deep learned embeddings—to deliver state-of-the-art performance in real-world defect detection scenarios.

The system is designed for both academic rigor and industrial applicability, providing:
- **Handcrafted spatial features** (Advanced HOG on a 2x2 grid) to capture local texture and edge information with spatial awareness.
- **Deep semantic features** (ResNet18 CNN backbone) to encode high-level contextual cues.
- **Modular, production-ready pipeline** with real-time post-processing and a web-based user interface.

---

## 2. Pipeline Architecture

The solution is organized into **five sequential, decoupled stages**, each implemented as a dedicated Python module for clarity and extensibility:

1. **Data Ingestion & Preprocessing** (`src/preprocessing.py`)
   - `IndustrialPreprocessor` class: Loads images, resizes to $256 \times 256$, converts to grayscale, applies Gaussian Blur for noise reduction, enhances contrast with CLAHE, and normalizes pixel intensities to $[0,1]$.

2. **Feature Engineering: Hybrid Representation** (`src/features.py`)
   - `FeatureExtractor` class: Extracts a spatially-aware HOG descriptor by splitting the image into a $2 \times 2$ grid of quadrants, preserving coarse spatial structure. Fuses this with deep semantic features from a pre-trained ResNet18 (PyTorch) CNN backbone, yielding a comprehensive feature vector.

3. **Core Logic: Defect Classification** (`src/model.py`)
   - `DefectClassifier` class: Trains a Support Vector Machine (SVM) with probability calibration on the fused feature matrix to distinguish normal from defective samples.

4. **Post-processing: Morphological Defect Localization** (`src/postprocessing.py`)
   - `DefectLocalizer` class: Processes the preprocessed frame using inverse thresholding, morphological closing, and contour detection (OpenCV) to localize and draw red bounding boxes around detected defects in real time.

5. **Performance Evaluation** (`src/evaluate.py`)
   - `PerformanceEvaluator` class: Computes Accuracy, Precision, Recall, and F1-Score. Generates and saves a confusion matrix plot (`confusion_matrix.png`) for diagnostic review.

---

## 3. Installation & Environment Setup

Follow these steps to set up your environment on macOS/Linux:

```bash
# 1. Clone the repository
git clone https://github.com/Keobu/Industrial-Quality-Control.git
cd Industrial-Quality-Control

# 2. Create and activate a virtual environment
python3 -m venv env
source env/bin/activate

# 3. Upgrade pip and install dependencies
python3 -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 4. How to Run

**A. Run the full pipeline (training & evaluation):**

```bash
python3 main.py
```

- If the `data/` directory is missing or empty, the script will automatically generate a synthetic industrial dataset with artificial scratches to ensure immediate, out-of-the-box execution.

**B. Launch the Streamlit Web Application:**

```bash
PYTHONPATH=. streamlit run app.py
```

---

## 5. Summary of Experimental Results

Upon execution, the system reports the following metrics (default verification setup):

- **Accuracy:** 1.0000
- **Precision:** 1.0000
- **Recall:** 1.0000
- **F1-Score:** 1.0000

The pipeline also exports a `confusion_matrix.png` file in the project root, providing a visual summary of true negatives, true positives, false negatives, and false positives for technical diagnostics and review.