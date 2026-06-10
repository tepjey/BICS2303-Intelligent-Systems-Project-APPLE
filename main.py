import os
import cv2
import numpy as np
import logging
import kagglehub
import tkinter as tk
from tkinter import filedialog, messagebox
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
# Using skimage for robust feature extraction
from skimage.feature import hog 

# Setup basic logging for diagnostics
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Load OpenCV's built-in face detector to clean up backgrounds
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# =======================================================
# 1. ADVANCED FEATURE EXTRACTION (HOG)
# =======================================================
def get_facial_features(img_48x48):
    """
    Return a consistent 1D HOG feature vector for an input image. The
    function is defensive: it accepts images that may not be exactly 48x48
    and will resize / convert as required.
    """
    img = np.asarray(img)

    # If image is colored, convert to grayscale by averaging channels
    if img.ndim == 3:
        # safe conversion in case channels != 3
        try:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        except Exception:
            img = img.mean(axis=2).astype(img.dtype)

    # Ensure correct shape (48x48)
    if img.size != 48 * 48:
        img = cv2.resize(img, (48, 48), interpolation=cv2.INTER_AREA)

    img = img.reshape(48, 48).astype('uint8')

    # HOG extracts structural shapes and gradients (wrinkles, eye/mouth shapes)
    # We set visualize=False so the function returns only the feature vector.
    features = hog(
        img,
        orientations=8,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=False,
        feature_vector=True,
        channel_axis=None,
    )

    return np.asarray(features, dtype=np.float64)

# =======================================================
# 2. LOAD DATASET & CLEAN DATA
# =======================================================
print("Downloading dataset from Kaggle...")
path = kagglehub.dataset_download("msambare/fer2013")

images = []
labels = []
# Kept standard mapping, but we will skip heavy noise if necessary
label_map = {'angry':0, 'disgust':1, 'fear':2, 'happy':3, 'sad':4, 'surprise':5, 'neutral':6}

print("Processing and cleaning images via HOG extraction...")
for subset in ['train', 'test']:
    sub_path = os.path.join(path, subset)
    for emotion in os.listdir(sub_path):
        e_path = os.path.join(sub_path, emotion)
        if not os.path.isdir(e_path): 
            continue
        lbl = label_map[emotion.lower()]
        
        # Limit disgust oversampling issues or skip entirely if you want quick balance
        count = 0
        for img_name in os.listdir(e_path):
            # Cap dataset size for standard ML training speed if needed (Optional)
            if count > 2500: break 
            
            img = cv2.imread(os.path.join(e_path, img_name), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                # Preprocessing: Histogram Equalization to normalize bad lighting
                # Ensure images are resized to 48x48 to keep consistent HOG inputs
                try:
                    resized_img = cv2.resize(img, (48, 48), interpolation=cv2.INTER_AREA)
                except Exception as ex:
                    logging.warning(f"Skipping image {img_name} due to resize error: {ex}")
                    continue

                img_equalized = cv2.equalizeHist(resized_img)

                images.append(img_equalized)
                labels.append(lbl)
                count += 1

X = np.array(images)
y = np.array(labels)

# Split data first to prevent data leakage
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Extracting spatial features...")
X_train_feat = np.array([get_facial_features(x) for x in X_train_raw])
X_test_feat = np.array([get_facial_features(x) for x in X_test_raw])

# Validate feature shapes and values (no NaNs, consistent vector lengths)
def validate_and_clean_features(Xf, yf, name="set"):
    clean_X = []
    clean_y = []
    for i, feat in enumerate(Xf):
        if feat is None:
            logging.warning(f"Dropping sample {i} from {name}: feature is None")
            continue
        feat = np.asarray(feat)
        if feat.ndim != 1:
            logging.warning(f"Dropping sample {i} from {name}: unexpected feature ndim={feat.ndim}")
            continue
        if np.isnan(feat).any() or np.isinf(feat).any():
            logging.warning(f"Dropping sample {i} from {name}: NaN or Inf in features")
            continue
        clean_X.append(feat)
        clean_y.append(yf[i])

    if len(clean_X) == 0:
        raise ValueError(f"No valid features remain in {name} after validation")

    return np.vstack(clean_X), np.array(clean_y)

# Clean training and test features
X_train_feat_clean, y_train = validate_and_clean_features(X_train_feat, y_train, name="train")
X_test_feat_clean, y_test = validate_and_clean_features(X_test_feat, y_test, name="test")

# Scale ONLY using training metrics
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_feat_clean)
X_test_scaled = scaler.transform(X_test_feat_clean)

# =======================================================
# 3. MODEL TRAINING & COMPARISON
# =======================================================
# Replaced slow Decision Trees with highly optimized SVM settings for spatial grids
models = {
    "Linear SVM": SVC(kernel='linear', class_weight='balanced', C=1.0, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=15, class_weight='balanced', random_state=42)
}

best_acc = 0
best_model = None
best_name = ""

print("\n--- Running Evaluation with Structured Features ---")
for name, clf in models.items():
    try:
        logging.info(f"Training {name} with X_train shape={X_train_scaled.shape}, y_train shape={y_train.shape}")
        clf.fit(X_train_scaled, y_train)
        preds = clf.predict(X_test_scaled)
    except Exception as ex:
        logging.error(f"Error training model {name}: {ex}")
        logging.error(f"X_train dtype={X_train_scaled.dtype}, X_train min/max={np.nanmin(X_train_scaled)}, {np.nanmax(X_train_scaled)}")
        logging.error(f"X_train shape={getattr(X_train_scaled, 'shape', None)}, y_train shape={getattr(y_train, 'shape', None)}")
        raise
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
# 4. TKINTER GUI PROTOTYPE (WITH AUTO-FACE DETECTION)
# =======================================================
class FERApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Group APPLE - Enhanced FER Prototype")
        self.root.geometry("450x330")
        
        self.emotions = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
        
        self.lbl_title = tk.Label(root, text="Facial Expression Classifier", font=("Arial", 14, "bold"))
        self.lbl_title.pack(pady=10)
        
        self.lbl_info = tk.Label(root, text=f"Active Backend: {best_name} ({best_acc*100:.1f}%)")
        self.lbl_info.pack()
        
        self.btn_load = tk.Button(root, text="Browse Image", command=self.upload_and_predict, width=20, height=2)
        self.btn_load.pack(pady=20)
        
        self.lbl_result = tk.Label(root, text="No image loaded.", font=("Arial", 12, "italic"), fg="gray")
        self.lbl_result.pack(pady=10)

    def upload_and_predict(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg")])
        if not file_path: 
            return
            
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            # Clean up the test image by looking for a face box first
            faces = face_cascade.detectMultiScale(img, 1.1, 4)
            if len(faces) > 0:
                # Crop to the active face frame to remove noisy backgrounds
                x, y, w, h = faces[0]
                img = img[y:y+h, x:x+w]
            
            resized = cv2.resize(img, (48, 48), interpolation=cv2.INTER_AREA)
            img_equalized = cv2.equalizeHist(resized)
            
            # Extract structured features
            feats = get_facial_features(img_equalized)
            scaled_feats = scaler.transform([feats])
            
        #     # Predict
        #     pred = best_model.predict(scaled_feats)[0]
        #     emotion_text = self.emotions[pred]
            
        #     color = "green" if emotion_text in ['Happy', 'Surprise', 'Neutral'] else "red"
        #     self.lbl_result.config(text=f"Predicted Expression: {emotion_text}", fg=color, font=("Arial", 12, "bold"))
        #     messagebox.showinfo("Result", f"Detected Emotion: {emotion_text}")
        # Predict
        pred = best_model.predict(scaled_feats)[0]

        # Use .get() with a fallback string to prevent KeyErrors
        emotion_text = self.emotions.get(int(pred), 'Unknown') 

        # Safely determine text color with a solid fallback
        if emotion_text in ['Happy', 'Surprise', 'Neutral']:
            color = "green"
        else:
            color = "red"

        # Update GUI safely
        self.lbl_result.config(text=f"Predicted Expression: {emotion_text}", fg=color, font=("Arial", 12, "bold"))
        messagebox.showinfo("Result", f"Detected Emotion: {emotion_text}")
if __name__ == "__main__":
    window = tk.Tk()
    app = FERApp(window)
    window.mainloop()