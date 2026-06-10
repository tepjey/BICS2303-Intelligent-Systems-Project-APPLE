import os
import cv2
import numpy as np
import kagglehub
import tkinter as tk
from tkinter import filedialog, messagebox
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from skimage.feature import hog, local_binary_pattern
# =======================================================
# 1. FEATURE EXTRACTION (EXPERT DOMAIN RATIOS)
# =======================================================


def get_facial_features(flat_pixels):
    img = flat_pixels.reshape(48, 48)
    
    # HOG: captures edge/gradient structure (shape of expressions)
    hog_features = hog(
        img,
        orientations=8,
        pixels_per_cell=(6, 6),
        cells_per_block=(2, 2),
        block_norm='L2-Hys'
    )
    
    # LBP: captures micro-texture (wrinkles, furrows)
    img_uint8 = (img * 255).astype(np.uint8)
    lbp = local_binary_pattern(img_uint8, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10), density=True)
    
    return np.concatenate([hog_features, lbp_hist])
# =======================================================
# 2. LOAD DATASET & SPLIT (NO DATA LEAKAGE)
# =======================================================
print("Downloading dataset from Kaggle...")
path = kagglehub.dataset_download("msambare/fer2013")

images = []
labels = []
label_map = {'angry':0, 'disgust':1, 'fear':2, 'happy':3, 'sad':4, 'surprise':5, 'neutral':6}

for subset in ['train', 'test']:
    sub_path = os.path.join(path, subset)
    for emotion in os.listdir(sub_path):
        e_path = os.path.join(sub_path, emotion)
        if not os.path.isdir(e_path): 
            continue
        lbl = label_map[emotion.lower()]
        
        for img_name in os.listdir(e_path):
            img = cv2.imread(os.path.join(e_path, img_name), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img.flatten().astype(np.float32) / 255.0)
                labels.append(lbl)

X = np.array(images)
y = np.array(labels)

# Split data first to prevent leakage
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Extract features on train and test separately
X_train_feat = np.array([get_facial_features(x) for x in X_train_raw])
X_test_feat = np.array([get_facial_features(x) for x in X_test_raw])

# Scale ONLY using training metrics
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_feat)
X_test_scaled = scaler.transform(X_test_feat)

# =======================================================
# 3. MODEL TRAINING & COMPARISON
# =======================================================
models = {
    "SVM": SVC(kernel='rbf', class_weight='balanced', C=2.0, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=12, class_weight='balanced', random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
}

best_acc = 0
best_model = None
best_name = ""

print("\n--- Running Comparative Analysis ---")
for name, clf in models.items():
    clf.fit(X_train_scaled, y_train)
    preds = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)
    
    print(f"\nModel: {name}")
    print(f"Accuracy: {acc*100:.2f}%")
    print(classification_report(y_test, preds))
    
    if acc > best_acc:
        best_acc = acc
        best_model = clf
        best_name = name

print(f"\nBest Model Selected: {best_name}")

# =======================================================
# 4. TKINTER GUI PROTOTYPE
# =======================================================
class FERApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Group APPLE - FER Prototype")
        self.root.geometry("450x300")
        
        self.emotions = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
        
        # Simple Labels
        self.lbl_title = tk.Label(root, text="Facial Expression Classifier", font=("Arial", 14, "bold"))
        self.lbl_title.pack(pady=10)
        
        self.lbl_info = tk.Label(root, text=f"Active Backend: {best_name} ({best_acc*100:.1f}%)")
        self.lbl_info.pack()
        
        # Button
        self.btn_load = tk.Button(root, text="Browse Image", command=self.upload_and_predict, width=20, height=2)
        self.btn_load.pack(pady=20)
        
        # Result Display Area
        self.lbl_result = tk.Label(root, text="No image loaded.", font=("Arial", 12, "italic"), fg="gray")
        self.lbl_result.pack(pady=10)

    def upload_and_predict(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg")])
        if not file_path: 
            return
            
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            resized = cv2.resize(img, (48, 48), interpolation=cv2.INTER_AREA)
            flat = resized.flatten().astype(np.float32) / 255.0
            
            # Feature engineering and scaling
            feats = get_facial_features(flat)
            scaled_feats = scaler.transform([feats])
            
            # Predict
            pred = best_model.predict(scaled_feats)[0]
            emotion_text = self.emotions[pred]
            
            # Simple color change
            color = "green" if emotion_text in ['Happy', 'Surprise', 'Neutral'] else "red"
            self.lbl_result.config(text=f"Predicted Expression: {emotion_text}", fg=color, font=("Arial", 12, "bold"))
            messagebox.showinfo("Result", f"Detected Emotion: {emotion_text}")

if __name__ == "__main__":
    window = tk.Tk()
    app = FERApp(window)
    window.mainloop()