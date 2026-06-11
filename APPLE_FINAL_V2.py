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
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
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
# 1. ALWAYS-REQUIRED DEPENDENCIES
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
# LANDMARK VISUALIZATION FUNCTION
# =======================================================

FEATURE_LINES = [
    (48, 54, (0,   255,   0), "Mouth Width"),
    (51, 57, (0,   220,   0), "Mouth Height"),
    (21, 39, (0,    60, 255), "L-Brow to Eye"),
    (22, 42, (0,   130, 255), "R-Brow to Eye"),
    (36, 39, (255, 200,   0), "L-Eye Width"),
    (42, 45, (255, 160,   0), "R-Eye Width"),
    (30, 51, (180,   0, 255), "Nose to Mouth"),
    (30,  8, (230,   0, 200), "Nose to Chin"),
    ( 0, 16, (255,   0, 180), "Jaw Width (norm)"),
]

def draw_landmark_visualization(image_path: str) -> np.ndarray | None:
    img  = cv2.imread(image_path)
    if img is None:
        return None

    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)

    if len(rects) == 0:
        scale = 2
        big   = cv2.resize(gray, (gray.shape[1]*scale, gray.shape[0]*scale))
        rects = detector(big, 1)
        if len(rects) == 0:
            return None
        shape_obj  = predictor(big, rects[0])
        pts = np.array([[shape_obj.part(i).x // scale,
                         shape_obj.part(i).y // scale] for i in range(68)])
    else:
        shape_obj = predictor(gray, rects[0])
        pts = np.array([[shape_obj.part(i).x, shape_obj.part(i).y] for i in range(68)])

    rect    = rects[0] if len(rects) > 0 else None
    overlay = img.copy()

    base      = min(img.shape[:2])
    dot_r     = max(2, base // 60)
    dot_outer = dot_r + 1
    ep_r      = max(3, base // 45)
    line_th   = max(1, base // 120)

    for x, y in pts:
        cv2.circle(img, (x, y), dot_r,    (255, 220, 50), -1)
        cv2.circle(img, (x, y), dot_outer, (0,   0,   0),  1)

    for (a, b, colour, label) in FEATURE_LINES:
        p1, p2 = tuple(pts[a]), tuple(pts[b])
        cv2.line(img, p1, p2, colour, line_th, cv2.LINE_AA)
        cv2.circle(img, p1, ep_r, (255, 255, 255), -1)
        cv2.circle(img, p2, ep_r, (255, 255, 255), -1)

    return img

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

    X_train_pixels = np.array([raw_images[i].flatten().astype(np.float32) for i in train_idx])
    X_test_pixels  = np.array([raw_images[i].flatten().astype(np.float32) for i in test_idx])
    scaler_pca     = MinMaxScaler()
    X_train_pixels = scaler_pca.fit_transform(X_train_pixels)
    X_test_pixels  = scaler_pca.transform(X_test_pixels)

    print("\nExtracting dlib landmark features (this will take several minutes)...")
    X_train_lm = np.array([get_landmark_features(raw_images[i]) for i in train_idx])
    X_test_lm  = np.array([get_landmark_features(raw_images[i]) for i in test_idx])
    scaler_lm  = MinMaxScaler()
    X_train_lm = scaler_lm.fit_transform(X_train_lm)
    X_test_lm  = scaler_lm.transform(X_test_lm)

    pca         = PCA(n_components=150, whiten=True, random_state=42)
    X_train_pca = pca.fit_transform(X_train_pixels)
    X_test_pca  = pca.transform(X_test_pixels)

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

    bundle_to_save = {
        "best_name": best_name, "best_acc": best_acc, "best_model": best_model,
        "best_uses_landmarks": best_uses_landmarks, "scaler_lm": scaler_lm,
        "scaler_pca": scaler_pca, "pca": pca, "label_map": label_map
    }
    with open(BUNDLE_PATH, 'wb') as f:
        pickle.dump(bundle_to_save, f)
    print("\n💾 System trained successfully and cached to 'fer_model_bundle.pkl'!")


# =======================================================
# 3. INTERACTIVE PRESENTATION GUI  (redesigned)
# =======================================================

# ── Palette ────────────────────────────────────────────
C_BG        = "#0D0F14"
C_PANEL     = "#1A1D26"
C_RAISED    = "#252A38"
C_ACCENT    = "#4FC3F7"
C_ACCENT2   = "#1E88E5"
C_TEXT      = "#E0E0E0"
C_MUTED     = "#6C757D"
C_BORDER    = "#2E3447"

# ── Per-emotion config ──────────────────────────────────
EMOTION_CFG = {
    "Angry":    {"emoji": "😠", "color": "#EF5350", "bg": "#2D1515"},
    "Disgust":  {"emoji": "🤢", "color": "#AB47BC", "bg": "#1E1228"},
    "Fear":     {"emoji": "😨", "color": "#FFA726", "bg": "#2B1D0A"},
    "Happy":    {"emoji": "😄", "color": "#66BB6A", "bg": "#0F2415"},
    "Sad":      {"emoji": "😢", "color": "#42A5F5", "bg": "#0A1929"},
    "Surprise": {"emoji": "😮", "color": "#FFCA28", "bg": "#2A2000"},
    "Neutral":  {"emoji": "😐", "color": "#90A4AE", "bg": "#1A1F26"},
}

class FERApp:
    THUMB_SIZE = 160

    def __init__(self, root):
        self.root = root
        self.root.title("Group APPLE — FER Prototype  |  BICS 2303")
        self.root.geometry("780x580")
        self.root.minsize(700, 520)
        self.root.configure(bg=C_BG)
        self._photo_ref = None
        self._scan_job  = None
        self._scan_step = 0
        self.show_viz   = tk.BooleanVar(value=True)

        self.emotion_labels = {
            0: "Angry", 1: "Disgust", 2: "Fear",
            3: "Happy", 4: "Sad",     5: "Surprise", 6: "Neutral",
        }

        self._build_ui()

    # ── UI construction ────────────────────────────────

    def _build_ui(self):
        root = self.root

        # ── Top header bar ──
        hdr = tk.Frame(root, bg=C_PANEL, pady=0)
        hdr.pack(fill="x")

        tk.Frame(hdr, bg=C_ACCENT, width=4).pack(side="left", fill="y")

        title_col = tk.Frame(hdr, bg=C_PANEL, padx=18, pady=12)
        title_col.pack(side="left", fill="both", expand=True)

        tk.Label(title_col,
                 text="Facial Expression Recognition",
                 font=("Segoe UI", 17, "bold"),
                 bg=C_PANEL, fg=C_TEXT).pack(anchor="w")
        tk.Label(title_col,
                 text="Group APPLE  ·  BICS 2303 Intelligent Systems",
                 font=("Segoe UI", 10),
                 bg=C_PANEL, fg=C_MUTED).pack(anchor="w")

        badge_frame = tk.Frame(hdr, bg=C_PANEL, padx=18)
        badge_frame.pack(side="right", fill="y")
        tk.Label(badge_frame,
                 text=f"{best_acc*100:.1f}%",
                 font=("Segoe UI", 22, "bold"),
                 bg=C_PANEL, fg=C_ACCENT).pack(anchor="e", pady=(10, 0))
        tk.Label(badge_frame,
                 text="test accuracy",
                 font=("Segoe UI", 8),
                 bg=C_PANEL, fg=C_MUTED).pack(anchor="e")
        tk.Label(badge_frame,
                 text=best_name,
                 font=("Segoe UI", 8, "italic"),
                 bg=C_PANEL, fg=C_MUTED).pack(anchor="e", pady=(0, 10))

        tk.Frame(root, bg=C_ACCENT, height=2).pack(fill="x")

        body = tk.Frame(root, bg=C_BG)
        body.pack(fill="both", expand=True, padx=20, pady=16)
        left = tk.Frame(body, bg=C_BG)
        left.pack(side="left", fill="y", padx=(0, 14))
        self.drop_frame = tk.Frame(left, bg=C_RAISED, width=self.THUMB_SIZE + 20,
                                   height=self.THUMB_SIZE + 20,
                                   highlightthickness=2,
                                   highlightbackground=C_BORDER)
        self.drop_frame.pack_propagate(False)
        self.drop_frame.pack(pady=(0, 12))

        self.thumb_label = tk.Label(self.drop_frame, bg=C_RAISED,
                                    text="📷\n\nno image\nloaded",
                                    font=("Segoe UI", 10),
                                    fg=C_MUTED, justify="center")
        self.thumb_label.place(relx=0.5, rely=0.5, anchor="center")

        self.btn_browse = tk.Button(
            left,
            text="  Browse Image",
            font=("Segoe UI", 11, "bold"),
            bg=C_ACCENT2, fg="white",
            activebackground=C_ACCENT,
            activeforeground="white",
            relief="flat", cursor="hand2",
            padx=14, pady=9,
            command=self.upload_and_predict,
        )
        self.btn_browse.pack(fill="x", pady=(0, 8))
        self._bind_hover(self.btn_browse, C_ACCENT, C_ACCENT2)

        chk_frame = tk.Frame(left, bg=C_RAISED, padx=10, pady=8,
                             highlightthickness=1,
                             highlightbackground=C_BORDER)
        chk_frame.pack(fill="x")
        self.chk = tk.Checkbutton(
            chk_frame,
            text=" Show landmark overlay",
            variable=self.show_viz,
            font=("Segoe UI", 9),
            bg=C_RAISED, fg=C_TEXT,
            activebackground=C_RAISED,
            activeforeground=C_ACCENT,
            selectcolor=C_RAISED,
            cursor="hand2",
        )
        self.chk.pack(anchor="w")

        right = tk.Frame(body, bg=C_BG)
        right.pack(side="left", fill="both", expand=True)

        self.scan_frame = tk.Frame(right, bg=C_BG)
        self.scan_frame.pack(fill="x", pady=(0, 10))

        self.scan_label = tk.Label(self.scan_frame, text="",
                                   font=("Segoe UI", 9), bg=C_BG, fg=C_ACCENT)
        self.scan_label.pack(anchor="w")

        style = ttk.Style()
        style.theme_use("default")
        style.configure("Scan.Horizontal.TProgressbar",
                        troughcolor=C_RAISED, background=C_ACCENT,
                        thickness=6, borderwidth=0)
        self.scan_bar = ttk.Progressbar(self.scan_frame,
                                        style="Scan.Horizontal.TProgressbar",
                                        mode="determinate", maximum=100)
        self.scan_bar.pack(fill="x")

        self.result_card = tk.Frame(right, bg=C_PANEL,
                                    highlightthickness=1,
                                    highlightbackground=C_BORDER)
        self.result_card.pack(fill="both", expand=True)

        self.idle_label = tk.Label(
            self.result_card,
            text="Load a face image\nto classify the emotion.",
            font=("Segoe UI", 12, "italic"),
            bg=C_PANEL, fg=C_MUTED, justify="center",
        )
        self.idle_label.place(relx=0.5, rely=0.5, anchor="center")

        self.res_emoji   = tk.Label(self.result_card, text="", font=("Segoe UI", 52),
                                    bg=C_PANEL)
        self.res_emotion = tk.Label(self.result_card, text="", font=("Segoe UI", 26, "bold"),
                                    bg=C_PANEL)
        self.res_model   = tk.Label(self.result_card, text="", font=("Segoe UI", 9),
                                    bg=C_PANEL, fg=C_MUTED)
        self.res_divider = tk.Frame(self.result_card, height=1, bg=C_BORDER)

        status_bar = tk.Frame(root, bg=C_PANEL, pady=5)
        status_bar.pack(fill="x", side="bottom")
        tk.Frame(status_bar, bg=C_ACCENT2, width=4).pack(side="left", fill="y")
        self.lbl_status = tk.Label(status_bar,
                                   text="Ready — browse an image to begin.",
                                   font=("Segoe UI", 9),
                                   bg=C_PANEL, fg=C_MUTED, padx=12)
        self.lbl_status.pack(side="left")

    # ── Hover helper ──────────────────────────────────

    def _bind_hover(self, widget, color_on, color_off):
        widget.bind("<Enter>", lambda e: widget.config(bg=color_on))
        widget.bind("<Leave>", lambda e: widget.config(bg=color_off))

    # ── Scan animation ─────────────────────────────────

    def _start_scan(self):
        self._scan_step = 0
        self.scan_bar["value"] = 0
        self.scan_label.config(text="Analysing…")
        self._animate_scan()

    def _animate_scan(self):
        self._scan_step += 7
        self.scan_bar["value"] = min(self._scan_step, 100)
        if self._scan_step < 100:
            self._scan_job = self.root.after(35, self._animate_scan)
        else:
            self.scan_label.config(text="Analysis complete")
            self.root.after(1200, self._clear_scan)

    def _clear_scan(self):
        self.scan_bar["value"] = 0
        self.scan_label.config(text="")

    # ── Thumbnail loader ───────────────────────────────

    def _load_thumbnail(self, file_path):
        try:
            pil_img   = Image.open(file_path).convert("RGB")
            pil_img   = pil_img.resize((self.THUMB_SIZE, self.THUMB_SIZE), Image.LANCZOS)
            self._photo_ref = ImageTk.PhotoImage(pil_img)
            self.thumb_label.config(image=self._photo_ref, text="", compound="center")
            self.drop_frame.config(highlightbackground=C_ACCENT)
        except Exception:
            pass   # non-fatal

    # ── Result display ─────────────────────────────────

    def _show_result(self, emotion_text):
        cfg   = EMOTION_CFG.get(emotion_text, {"emoji": "🤔", "color": C_TEXT, "bg": C_PANEL})
        color = cfg["color"]
        bg    = cfg["bg"]

        # fade card background
        self.result_card.config(bg=bg, highlightbackground=color)
        self.idle_label.place_forget()

        self.res_divider.pack_forget()
        self.res_emoji.pack_forget()
        self.res_emotion.pack_forget()
        self.res_model.pack_forget()

        self.res_emoji.config(text=cfg["emoji"], bg=bg)
        self.res_emotion.config(text=emotion_text, bg=bg, fg=color)
        self.res_model.config(bg=bg,
                               text=f"Model: {best_name}  ·  processed at 48 × 48 px")

        self.res_divider.config(bg=color)
        self.res_divider.pack(fill="x", padx=30, pady=(0, 6))
        self.res_emoji.pack(pady=(20, 4))
        self.res_emotion.pack()
        self.res_model.pack(pady=(6, 0))

    # ── Core pipeline ──────────────────────────────────

    def preprocess_image(self, file_path):
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        h, w    = img.shape
        min_dim = min(h, w)
        sy, sx  = (h - min_dim) // 2, (w - min_dim) // 2
        return cv2.equalizeHist(
            cv2.resize(img[sy:sy + min_dim, sx:sx + min_dim],
                       (48, 48), interpolation=cv2.INTER_AREA)
        )

    def upload_and_predict(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.png *.jpg *.jpeg")]
        )
        if not file_path:
            return

        self._load_thumbnail(file_path)
        self._start_scan()
        self.lbl_status.config(text=f"Loaded: {os.path.basename(file_path)}", fg=C_MUTED)

        img_48 = self.preprocess_image(file_path)
        if img_48 is None:
            self.lbl_status.config(text="⚠  Could not load image.", fg="#EF5350")
            return

        if best_uses_landmarks:
            feats  = get_landmark_features(img_48)
            scaled = scaler_lm.transform([feats])
        else:
            flat   = img_48.flatten().astype(np.float32)
            scaled = pca.transform(scaler_pca.transform([flat]))

        pred         = best_model.predict(scaled)[0]
        emotion_text = self.emotion_labels[pred]

        self._show_result(emotion_text)
        self.lbl_status.config(
            text=f"✔  Classified as '{emotion_text}'  ·  {os.path.basename(file_path)}",
            fg=EMOTION_CFG.get(emotion_text, {}).get("color", C_TEXT)
        )

        if self.show_viz.get():
            annotated = draw_landmark_visualization(file_path)

            if annotated is not None:
                h, w    = annotated.shape[:2]
                min_dim = 600
                max_w   = 1200

                if min(h, w) < min_dim:
                    scale     = min_dim / min(h, w)
                    annotated = cv2.resize(annotated,
                                           (int(w * scale), int(h * scale)),
                                           interpolation=cv2.INTER_CUBIC)
                    h, w = annotated.shape[:2]

                if w > max_w:
                    scale     = max_w / w
                    annotated = cv2.resize(annotated,
                                           (max_w, int(h * scale)),
                                           interpolation=cv2.INTER_AREA)

                win_title = f"Landmark Overlay  |  {emotion_text}  — Group APPLE"
                cv2.imshow(win_title, annotated)
                cv2.waitKey(1)
                self.lbl_status.config(
                    text=f"✔  '{emotion_text}'  ·  landmark window open (press any key to close it)",
                    fg=EMOTION_CFG.get(emotion_text, {}).get("color", C_TEXT)
                )
            else:
                self.lbl_status.config(
                    text="⚠  No face detected in image — landmark overlay skipped.",
                    fg="#FFA726"
                )


if __name__ == "__main__":
    window = tk.Tk()
    app    = FERApp(window)
    window.mainloop()
    cv2.destroyAllWindows()
