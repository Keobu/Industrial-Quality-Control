# Industrial Quality Control - Automated Defect Detection

An end-to-end Computer Vision pipeline designed for automated surface inspection and defect detection in manufacturing environments. This repository implements a hybrid approach combining classical handcrafted features and Deep Learning representations to deliver robust and efficient anomaly detection.

## 🚀 Project Overview

In industrial manufacturing, identifying surface defects (such as cracks, scratches, or structural deformations) is critical to ensuring product quality. This project addresses the problem by constructing a complete, modular vision pipeline that handles everything from raw image ingestion to performance evaluation.

### Key Highlights:
* **Hybrid Representation**: Fuses spatial/geometric high-frequency features (HOG) with high-level semantic deep features (ResNet18).
* **Robust Preprocessing**: Mitigates factory lighting variations using adaptive histogram equalization (CLAHE).
* **Production-Ready Core**: Employs a Support Vector Machine (SVM) classifier for fast inference times suitable for real-time edge deployment.

---

## 🛠️ Pipeline Architecture

The application is structured into four sequential, decoupled stages as required by production-level standards:

1. **Data Acquisition & Preprocessing (`src/preprocessing.py`)**: 
   * Image loading and resizing to a unified $256 \times 256$ resolution.
   * Grayscale conversion and Gaussian Blurring for high-frequency sensor noise reduction.
   * Contrast Limited Adaptive Histogram Equalization (CLAHE) to enhance micro-defects under non-uniform illumination.
   * Pixel intensity normalization to the $[0, 1]$ range.

2. **Feature Engineering (`src/features.py`)**:
   * **Classical Features**: Extracts Histogram of Oriented Gradients (HOG) to capture local shape, edge orientations, and fine scratch textures.
   * **Learned Features**: Utilizes a pre-trained ResNet18 backbone (stripped of its fully connected classification layer) to extract global contextual semantic embeddings.
   * **Feature Fusion**: Concatenates both feature vectors into a comprehensive unified matrix representation.

3. **Core Logic (`src/model.py`)**:
   * Trains a Support Vector Machine (SVM) with a linear/RBF kernel on the fused feature matrix to separate normal samples from structural anomalies.

4. **Performance Evaluation (`src/evaluate.py`)**:
   * Computes Accuracy, Precision, Recall, and F1-Score.
   * Automatically generates and saves a visual Confusion Matrix plot (`confusion_matrix.png`).

---

git clone [https://github.com/Keobu/Industrial-Quality-Control.git](https://github.com/Keobu/Industrial-Quality-Control.git)
## ⚙️ Setup & Usage

### 1. Clone the repository
```bash
git clone https://github.com/Keobu/Industrial-Quality-Control.git
cd Industrial-Quality-Control
```

### 2. Create and activate a virtual environment (recommended)
```bash
python3 -m venv env
source env/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the pipeline
If the `data/` folder does not contain a dataset, the system will automatically generate a synthetic one for immediate testing.
```bash
python3 main.py
```

### 📊 Results
At the end of execution, the main metrics (Accuracy, Precision, Recall, F1-Score) will be printed to the console, and a `confusion_matrix.png` image will be saved in the root directory for technical review.
The generated confusion_matrix.png is saved in the root directory to document true negatives, true positives, false negatives, and false positives for the technical review.