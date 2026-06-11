# Facial Expression Recognition (FER) Interactive System Topology

## Course Details
* **Course:** BICS 2303: Intelligent Systems (Semester 2 2025/2026)
* **Institution:** International Islamic University Malaysia (IIUM)
* **Lecturer:** Asst. Prof. Dr. Dini Oktarina Dwi Handayani
* **Group Name:** Group APPLE
* **Submission Date:** 12th June 2026

## Group Members
* Ahmad Mukhlis bin Zakariah (2514371)
* Muhammad A'thif Uzair bin Shaedin (2514847)
* Zain Kadin bin Mohd Fadzil Arsady (2517135)
* Muhammad Aiman bin Sufian (2511349)
* Hidayat bin Mohd Najib (2516335)

---

## 1. Project Overview
This project delivers an interactive Intelligent System prototype that evaluates multi-class human facial emotion classification across seven distinct target categories: **Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral**. 

To overcome the spatial limitations of raw, flattened pixel configurations (Branch A), the system integrates an Expert Domain Approach (Branch B). This branch extracts 68-point dlib facial landmarks to derive scale-invariant spatial geometric ratios. The framework includes an automated fallback loop utilizing pixel sub-region statistical distributions to preserve runtime stability under poor lighting conditions. 

The application implements a strict execution workflow where feature scalers are computed strictly after data partitioning to neutralize data leakage. It trains and compares three distinct machine learning topologies: Support Vector Machines (SVM), Decision Trees, and Random Forest ensembles. To maximize classroom demonstration efficiency, the application features an integrated persistence layer that caches the optimal architecture to disk, allowing for sub-millisecond graphical user interface (GUI) inference.

---

## 2. System Architecture & Features
* **Dual-Branch Preprocessing:** Supports global high-dimensional pixel reduction via Principal Component Analysis (150 Principal Components) alongside localized 68-point geometric landmark vectorization.
* **Scale Invariance:** Normalizes geometric distances using the Inter-Ocular/Jawline distance array to handle user-to-lens proximity variations.
* **Defensive Robustness:** Provides a statistical matrix zone tracker (brows, eyes, nose, mouth arrays) to keep prediction tasks online if dlib's frontal face detector cannot isolate a face boundary.
* **Serialized Model Persistence:** Incorporates smart caching (`fer_model_bundle.pkl`). It performs heavy multi-model training and plot generation exactly once, then loads the bundle instantly for live presentations.
* **Interactive GUI Dashboard:** Built using Python's native Tkinter package, allowing seamless image uploading, real-time feature extraction, and automated color-coded target display outputs.

---

## 3. Package Dependencies & Prerequisites
The prototype runs on standard Python environments (Python 3.10 - 3.12). The required third-party libraries must be available in your runtime system:

```bash
pip install kagglehub opencv-python numpy dlib scikit-learn matplotlib seaborn pillow

```

### Essential Local Assets

To unlock shape predictor regression tree mapping, dlib requires its pre-trained weights file.

* **File Name:** `shape_predictor_68_face_landmarks.dat`
* **Automated Behavior:** If this asset is missing from the directory root, the system's script will automatically launch an online download sequence using `urllib` to fetch it safely (~100 MB).

---

## 4. Repository Structure

Ensure your final workspace submission directory follows this exact structural layout to facilitate smooth execution:

```
├── APPLE_FINAL.py                        # Consolidated Main Source Application
├── shape_predictor_68_face_landmarks.dat   # dlib 68-Point Weights File (Auto-downloaded if missing)
├── fer_model_bundle.pkl                   # Serialized Model & Scaler Deployment Bundle
├── accuracy_comparison.png                # Generated Algorithmic Comparison Bar Chart
├── confusion_matrix.png                   # Generated Evaluation Heatmap Plot
└── README.md                              # Operational Guide Documentation

```

---

## 5. Execution Instructions

### Step 1: Initialize the Machine Learning Pipeline

Open your terminal inside the project folder root and run the master script.

```bash
python APPLE_FINAL_V2.py

```

* **First-Time Initialization Behavior:** If no local cache (`fer_model_bundle.pkl`) is detected, the framework will download the complete [FER2013](https://www.kaggle.com/datasets/msambare/fer2013) dataset from the Kaggle repository hub. It splits the data, extracts landmark mappings, saves performance charts to your directory, and writes out the model bundle cache. **Note: This complete baseline calibration phase takes several minutes.**
* **Cached Presentation Launch:** On all subsequent runs, the script will skip the training lifecycle completely, find the cache, print the confirmation logs, and boot up your Tkinter desktop application interface in under 2 seconds.

### Step 2: Operating the Presentation Dashboard

1. Once the graphical frame appears, click the **📂 Browse Image** button.
2. Select any target facial expression sample (`.png`, `.jpg`, or `.jpeg`).
3. The dashboard will process the crop configurations at a native $48\times48\text{px}$ boundary matrix, feed the data across the active presenter ensemble model, open an informative system alert dialog box, and update the display label text using optimized conditional green/red alert color tracking.

---

## 6. Algorithmic Comparison Performance Metrics

The framework automatically logs precision, recall, and macro average metrics across your terminal during its evaluation cycle:

* **Random Forest Classifier (Active Landmark Presenter):** **44.41% Accuracy**
* *Strengths:* Excellent structural mapping on Happy (F1: 0.67) and Surprise (F1: 0.51). High ensemble voting reliability across structured geometric ratios.


* **Support Vector Machine Baseline (PCA Branch):** **39.54% Accuracy**
* *Strengths:* Solid global variance mapping, but structurally limited on ambiguous boundaries.


* **Decision Tree Classifier (Landmark Branch):** **35.15% Accuracy**
* *Limitations:* Overfitting risks handled through depth bounds, but suffers from single tree split constraints.



---

## 7. Troubleshooting Notes

* **dlib Compilation Failure:** If installing `dlib` via `pip` fails, ensure your operating environment has CMake installed (`pip install cmake`) and has access to Visual Studio C++ build tools.
* **Tkinter Canvas Geometry Errors:** The window coordinates are bound tightly at a non-resizable $500\times380\text{px}$ grid block to prevent cross-platform visual element tearing.
* **Image Loading Check:** If an uploaded target image fails to load, ensure the image file path does not contain unexpected non-ASCII special characters or unreadable folder sub-string indices.
