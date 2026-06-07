# Facial Expression Recognition via Regional Geometric Ratio Feature Engineering

## 👥 Group Members (APPLE)
* Ahmad Mukhlis bin Zakariah (2514371)
* Muhammad A'thif Uzair bin Shaedin (2514847)
* Zain Kadin bin Mohd Fadzil Arsady (2517135)
* Muhammad Aiman bin Sufian (2511349)
* Hidayat bin Mohd Najib (2516335)

---

## 📌 Project Overview
This repository contains the intelligent system project developed by **Group APPLE** for the course **BICS 2303: Intelligent Systems (Semester 2 2025/2026)** at the **International Islamic University Malaysia (IIUM)** under the supervision of **Asst. Prof. Dr. Dini Oktarina Dwi Handayani**.

The system addresses critical challenges in traditional Facial Expression Recognition (FER) workflows by shifting away from linear, raw-pixel dimensional reduction (such as PCA on flat matrices) to an **Expert Domain Approach** using hand-crafted regional geometric ratios. The framework implements a strict data-separation pipeline to completely prevent data leakage and compares performance metrics across three machine learning topologies:
1. Support Vector Machine (SVC with RBF Kernel)
2. Decision Tree Classifier
3. Random Forest Classifier (Ensemble Topology)

The highest-performing model is deployed into a functional, user-friendly desktop application prototype built with Python's native `tkinter` GUI library.

---

## 🔬 System Architecture & Pipeline
To maintain full data integrity and prevent any mathematical bleeding of evaluation statistics into the training parameters, the codebase follows a strict linear sequence:

1. **Stratified Partitioning:** Splitting raw facial arrays into an 80% training set and 20% validation set, stratifying by target labels to guarantee stable representation of rare classes (e.g., *Disgust*).
2. **Regional Anatomical Slicing:** Reshaping flat gray pixel arrays back into spatial matrices to isolate explicit horizontal zones (Eyebrows, Eyes, Nose Bridge, and Mouth) along with vertical symmetry zones.
3. **Geometric Feature Extraction:** Calculating eight distinct localized ratios tracking texture variance, intensity deltas, asymmetry, and muscle contractions.
4. **Isolated Scaling:** Fitting a standard Z-score normalizer strictly on training data metrics, then down-transforming validation splits separately.
5. **Ensemble Classification:** Evaluating and optimizing hyperparameter bounds across all models using balanced class distributions to eliminate algorithmic bias.

---

## 📊 Dataset Profile: FER2013
The model is validated against the benchmark [FER2013](https://www.kaggle.com/datasets/msambare/fer2013) open-source dataset, comprised of 35,887 grayscale facial images restricted to a native $48\times48$ pixel resolution. The data is structurally categorized into seven primary targets:
* `0`: Angry
* `1`: Disgust *(Severe Underrepresentation)*
* `2`: Fear
* `3`: Happy *(Dominant Majority)*
* `4`: Sad
* `5`: Surprise
* `6`: Neutral

---

## 💻 Installation & Dependencies

Ensure you have Python 3.8+ installed locally. Clone the repository and install the verified baseline libraries via `pip`:

```bash
pip install opencv-python scikit-learn numpy kagglehub
```

### Core Prerequisites

* `opencv-python`: Used for target file read operations, matrix manipulation, and downsampling interpolation.
* `scikit-learn`: Powers our data splitting, scaling vectors, cross-evaluation reports, and baseline classification topologies.
* `numpy`: Coordinates spatial zone array slicing and high-speed algebraic ratio extractions.
* `kagglehub`: Automates secure dataset downloading directly from official Kaggle repositories.

---

## 🚀 Running the Project

Run the fully integrated pipeline using your local terminal window:

```bash
python main.py
```

### Script Workflow Execution

1. The script initializes the download connection to fetch the FER2013 data from Kaggle (featuring an automated synthetic mock matrix fallback generator if network firewalls block the API request).
2. The model pipeline auto-runs to fit, evaluate, and output detailed comparative performance analysis matrices directly inside your console window.
3. The native OS `tkinter` window opens. Use the **"Browse Image"** interface button to test the deployed model using external image paths.
