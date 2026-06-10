import kagglehub
import os
import cv2
import numpy as np
import urllib.request
import dlib
import pickle
import matplotlib
matplotlib.use('Agg')   
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import filedialog, messagebox
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, accuracy_score, confusion_matrix)

# =======================================================
# CONFIGURATION & PERSISTENCE PATHS
# =======================================================

BUNDLE_PATH = "fer_model_bundle.pkl"
MODEL_PATH  = "shape_predictor_68_face_landmarks.dat"
MODEL_URL   = (
    "https://github.com/tzutalin/dlib-android/raw/master/"
    "data/shape_predictor_68_face_landmarks.dat"
)

# =======================================================
# 1. ALWAYS-REQUIRED DEPENDENCIES (For Live GUI Inference)
# =======================================================

if not os.path.exists(MODEL_PATH):
    print("Downloading dlib 68-point shape predictor model (~100 MB)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model downloaded successfully.")

detector  = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(MODEL_PATH)

def euclidean(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def shape_to_array(shape):
    return np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)])

def get_landmark_features(gray_img_48x48):
    img_96 = cv2.resize(gray_img_48x48, (96, 96))
    faces  = detector(img_96, 0)

    if len(faces) > 0:
        pts = shape_to_array(predictor(img_96, faces[0]))
        jaw_left, jaw_right = pts[0], pts[16]
        chin, nose_tip      = pts[8], pts[30]
        left_eye_outer, left_eye_inner   = pts[36], pts[39]
        right_eye_outer, right_eye_inner = pts[45], pts[42]
        left_brow_inner, right_brow_inner = pts[21], pts[22]
        mouth_left, mouth_right           = pts[48], pts[54]
        mouth_top, mouth_bottom           = pts[51], pts[57]

        norm = euclidean(jaw_left, jaw_right) + 1e-6

        return [
            euclidean(mouth_left, mouth_right) / norm,
            euclidean(mouth_top, mouth_bottom) / norm,
            euclidean(left_brow_inner, left_eye_inner) / norm,
            euclidean(right_brow_inner, right_eye_inner) / norm,
            euclidean(left_eye_outer, left_eye_inner) / norm,
            euclidean(right_eye_outer, right_eye_inner) / norm,
            euclidean(nose_tip, mouth_top) / norm,
            euclidean(nose_tip, chin) / norm,
            euclidean(mouth_left, mouth_right) / (euclidean(mouth_top, mouth_bottom) + 1e-6)
        ]
    else:
        img      = gray_img_48x48.astype(np.float32) / 255.0
        brows    = img[10:20, :];  eyes  = img[15:25, :]
        nose     = img[20:30, :];  mouth = img[30:45, :]
        img_mean = np.mean(img) + 1e-6
        img_std  = np.std(img)  + 1e-6
        return [
            np.std(mouth) / (np.std(eyes) + 1e-6),
            np.mean(brows) / (np.mean(mouth) + 1e-6),
            (np.max(mouth) - np.min(mouth)) / img_mean,
            np.std(eyes) / img_std,
            np.mean(nose) / (np.mean(mouth) + 1e-6),
            np.std(mouth[:, :24]) / (np.std(mouth[:, 24:]) + 1e-6),
            abs(np.mean(brows) - np.mean(eyes)) / img_std,
            np.var(brows) / (np.var(mouth) + 1e-6),
            0.0
        ]

# =======================================================
# 2. SMART CONDITIONAL EXECUTION LOOP
# =======================================================

if os.path.exists(BUNDLE_PATH):
    print("\n" + "="*50)
    print("PRE-TRAINED CACHE DETECTED!")
    print("="*50)
    with open(BUNDLE_PATH, 'rb') as f:
        bundle = pickle.load(f)
        
    best_name           = bundle["best_name"]
    best_acc            = bundle["best_acc"]
    best_model          = bundle["best_model"]
    best_uses_landmarks = bundle["best_uses_landmarks"]
    scaler_lm           = bundle["scaler_lm"]
    scaler_pca          = bundle["scaler_pca"]
    pca                 = bundle["pca"]
    label_map           = bundle["label_map"]
    
    print(f"Loaded Active Presenter Model: {best_name} (Accuracy: {best_acc*100:.2f}%)")
else:
    print("\n" + "="*50)
    print("❄️ NO CACHE FOUND. RUNNING COMPREHENSIVE TRAINING PIPELINE ONCE... ❄️")
    print("="*50)
    
    # --- Dataset Loading ---
    print("Downloading FER2013 dataset from Kaggle...")
    path = kagglehub.dataset_download("msambare/fer2013")

    raw_images, labels = [], []
    label_map = {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'sad': 4, 'surprise': 5, 'neutral': 6}

    for subset in ['train', 'test']:
        s_path = os.path.join(path, subset)
        for emotion in os.listdir(s_path):
            e_path = os.path.join(s_path, emotion)
            if not os.path.isdir(e_path): continue
            lbl = label_map.get(emotion.lower(), -1)
            if lbl == -1: continue
            for img_name in os.listdir(e_path):
                img = cv2.imread(os.path.join(e_path, img_name), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    raw_images.append(img)
                    labels.append(lbl)

    raw_images = np.array(raw_images, dtype=np.uint8)
    y = np.array(labels)

    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42, stratify=y)
    y_train, y_test = y[train_idx], y[test_idx]

    # --- Processing Branch A (Pixels) ---
    X_train_pixels = np.array([raw_images[i].flatten().astype(np.float32) for i in train_idx])
    X_test_pixels  = np.array([raw_images[i].flatten().astype(np.float32) for i in test_idx])
    scaler_pca     = MinMaxScaler()
    X_train_pixels = scaler_pca.fit_transform(X_train_pixels)
    X_test_pixels  = scaler_pca.transform(X_test_pixels)

    # --- Processing Branch B (Landmarks) ---
    print("\nExtracting dlib landmark features (this will take several minutes)...")
    X_train_lm = np.array([get_landmark_features(raw_images[i]) for i in train_idx])
    X_test_lm  = np.array([get_landmark_features(raw_images[i]) for i in test_idx])
    scaler_lm  = MinMaxScaler()
    X_train_lm = scaler_lm.fit_transform(X_train_lm)
    X_test_lm  = scaler_lm.transform(X_test_lm)

    # --- PCA Engineering ---
    pca         = PCA(n_components=150, whiten=True, random_state=42)
    X_train_pca = pca.fit_transform(X_train_pixels)
    X_test_pca  = pca.transform(X_test_pixels)

    # --- Comparative Model Block ---
    EMOTION_NAMES = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']
    models = {
        "SVM (PCA + Eigenface)": {"clf": SVC(kernel='rbf', class_weight='balanced', C=5, gamma=0.001, random_state=42), "X_tr": X_train_pca, "X_te": X_test_pca},
        "Decision Tree (Landmark)": {"clf": DecisionTreeClassifier(max_depth=10, min_samples_leaf=8, class_weight='balanced', random_state=42), "X_tr": X_train_lm, "X_te": X_test_lm},
        "Random Forest (Landmark)": {"clf": RandomForestClassifier(n_estimators=150, max_depth=12, min_samples_leaf=5, class_weight='balanced', random_state=42), "X_tr": X_train_lm, "X_te": X_test_lm},
    }

    best_acc, best_model, best_name, best_preds = 0, None, "", None
    results = {}

    for name, bundle_m in models.items():
        clf = bundle_m["clf"]
        clf.fit(bundle_m["X_tr"], y_train)
        preds = clf.predict(bundle_m["X_te"])
        acc   = accuracy_score(y_test, preds)
        results[name] = acc * 100

        print(f"\nModel   : {name}\nAccuracy: {acc * 100:.2f}%\n", classification_report(y_test, preds, target_names=EMOTION_NAMES))

        if acc > best_acc:
            best_acc, best_model, best_name, best_preds = acc, clf, name, preds
            best_uses_landmarks = "Landmark" in name

    # --- Generate Plots Once ---
    cm = confusion_matrix(y_test, best_preds)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=EMOTION_NAMES, yticklabels=EMOTION_NAMES, ax=ax)
    ax.set_title(f'Confusion Matrix — {best_name}', fontsize=13, fontweight='bold')
    plt.tight_layout(); plt.savefig('confusion_matrix.png', dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(results.keys(), results.values(), color=['#4C72B0', '#DD8452', '#55A868'], width=0.5)
    for bar, val in zip(bars, results.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8, f'{val:.2f}%', ha='center', fontweight='bold')
    ax.set_xticks(range(len(results))); ax.set_xticklabels(list(results.keys()), fontsize=10)
    plt.tight_layout(); plt.savefig('accuracy_comparison.png', dpi=150); plt.close()

    # --- Cache the Deployment Bundle to Disk ---
    bundle_to_save = {
        "best_name": best_name, "best_acc": best_acc, "best_model": best_model,
        "best_uses_landmarks": best_uses_landmarks, "scaler_lm": scaler_lm,
        "scaler_pca": scaler_pca, "pca": pca, "label_map": label_map
    }
    with open(BUNDLE_PATH, 'wb') as f:
        pickle.dump(bundle_to_save, f)
    print("\n💾 System trained successfully and cached to 'fer_model_bundle.pkl'!")

# =======================================================
# 3. INTERACTIVE PRESENTATION GUI
# =======================================================

class FERApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Group APPLE — FER Prototype (BICS 2303)")
        self.root.geometry("500x380")
        self.root.resizable(False, False)
        self.root.configure(bg="#f0f0f0")

        emotion_display = {
            0: 'Angry', 1: 'Disgust', 2: 'Fear',
            3: 'Happy', 4: 'Sad',    5: 'Surprise', 6: 'Neutral'
        }
        self.emotion_labels = emotion_display

        tk.Label(root, text="Facial Expression Recognition", font=("Arial", 15, "bold"), bg="#f0f0f0").pack(pady=(18, 2))
        tk.Label(root, text="Group APPLE  |  BICS 2303 Intelligent Systems", font=("Arial", 10), bg="#f0f0f0", fg="#555555").pack()
        tk.Label(root, text=f"Active Model: {best_name}  |  Test Accuracy: {best_acc*100:.1f}%", font=("Arial", 10, "italic"), bg="#f0f0f0", fg="#333333").pack(pady=(4, 0))
        tk.Frame(root, height=2, bg="#cccccc").pack(fill='x', padx=20, pady=10)

        tk.Button(root, text="📂  Browse Image", command=self.upload_and_predict, font=("Arial", 11), width=22, height=2, bg="#4C72B0", fg="white", relief="flat", activebackground="#3a5a8a").pack(pady=8)
        self.lbl_result = tk.Label(root, text="Load a facial image to begin classification.", font=("Arial", 12, "italic"), fg="#888888", bg="#f0f0f0", wraplength=440)
        self.lbl_result.pack(pady=14)
        self.lbl_note = tk.Label(root, text="", font=("Arial", 9), fg="#777777", bg="#f0f0f0")
        self.lbl_note.pack()

    def preprocess_image(self, file_path):
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if img is None: return None
        h, w = img.shape
        min_dim = min(h, w)
        sy, sx = (h - min_dim) // 2, (w - min_dim) // 2
        return cv2.equalizeHist(cv2.resize(img[sy:sy + min_dim, sx:sx + min_dim], (48, 48), interpolation=cv2.INTER_AREA))

    def upload_and_predict(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg")])
        if not file_path: return

        img_48 = self.preprocess_image(file_path)
        if img_48 is None:
            messagebox.showerror("Error", "Could not load image.")
            return

        if best_uses_landmarks:
            feats  = get_landmark_features(img_48)
            scaled = scaler_lm.transform([feats])
        else:
            flat   = img_48.flatten().astype(np.float32)
            scaled = pca.transform(scaler_pca.transform([flat]))

        pred = best_model.predict(scaled)[0]
        emotion_text = self.emotion_labels[pred]

        color = "#2a7a2a" if emotion_text in ['Happy', 'Surprise', 'Neutral'] else "#b22222"
        self.lbl_result.config(text=f"Predicted Expression:  {emotion_text}", fg=color, font=("Arial", 13, "bold"))
        self.lbl_note.config(text=f"Model: {best_name}  |  Processed at 48×48px")
        messagebox.showinfo("Classification Result", f"Detected Emotion: {emotion_text}\n\nModel: {best_name}")

if __name__ == "__main__":
    window = tk.Tk()
    app    = FERApp(window)
    window.mainloop()