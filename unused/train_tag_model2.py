import os
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
import joblib

# ==========================
# 0) CONFIG
# ==========================
CSV_PATH   = r"C:\Users\Jonas\Downloads\EnTagging\csv\summary.csv"  # <- anpassen
TEXT_COL   = "text"
TAGS_COL   = "tags"
TITLE_COL  = "title"   # falls es die Spalten nicht gibt, wird leer aufgefüllt
AUTHOR_COL = "author"

MIN_FREQ   = 10        # Mindesthäufigkeit je Label
TITLE_BOOST = 2        # Titel 2x anhängen, wirkt wie höheres Gewicht
AUTHOR_PREFIX = "__AUTHOR_"  # Autor als Feature-Token

# Labels zusammenlegen (Beispiel!)
MERGE = {
    # "Whg 1 Karl-Pfaff-Straße": "Whg Karl-Pfaff-Straße",
    # "Whg 2 Karl-Pfaff-Straße": "Whg Karl-Pfaff-Straße",
    # "Whgn Karl-Pfaff-Straße":  "Whg Karl-Pfaff-Straße",

    # weitere Synonyme hier sammeln …
    # "HV Ludwigsburger HV": "HV Ludwigsburger",
    # "HV Matthias Götz GmbH": "HV Götz",
}

# ==========================
# 1) CSV einlesen & vorbereiten
# ==========================
df = pd.read_csv(CSV_PATH)

# fehlende Spalten auffüllen
for col in [TEXT_COL, TAGS_COL]:
    if col not in df.columns:
        df[col] = ""
for col in [TITLE_COL, AUTHOR_COL]:
    if col not in df.columns:
        df[col] = ""

# NaNs -> ""
for col in [TEXT_COL, TITLE_COL, AUTHOR_COL, TAGS_COL]:
    if col in df.columns:
        df[col] = df[col].fillna("")

def split_tags(x):
    return [t.strip() for t in str(x).split(",") if t.strip()]

df["tag_list"] = df[TAGS_COL].apply(split_tags)

# ==========================
# 2) Label-Normalisierung & -Filter
# ==========================
def normalize_label(t):
    return MERGE.get(t, t)

# Mergen
df["tag_list"] = df["tag_list"].apply(lambda tags: [normalize_label(t) for t in tags])

# Häufigkeiten zählen (nach Merge)
cnt = Counter(t for tags in df["tag_list"] for t in tags)
# seltene Labels entfernen
df["tag_list"] = df["tag_list"].apply(lambda tags: [t for t in tags if cnt[t] >= MIN_FREQ])
# Zeilen ohne verbleibende Labels raus
df = df[df["tag_list"].map(len) > 0].reset_index(drop=True)

print(f"[Info] Datensätze nach Label-Filter: {len(df)}; unterschiedliche Labels: {len(set(t for tags in df['tag_list'] for t in tags))}")

# ==========================
# 3) Feature-Engineering (Titel/Autor leicht boosten)
# ==========================
def build_text(row):
    text   = str(row[TEXT_COL] or "")
    title  = str(row[TITLE_COL] or "").strip()
    author = str(row[AUTHOR_COL] or "").strip()

    parts = [text]
    if title:
        parts.append((" " + title) * TITLE_BOOST)
    if author:
        # Autor als robustes Token (vermeidet Namens-Varianten in Wort-Ngrammen)
        parts.append(f"{AUTHOR_PREFIX}{author.lower().replace(' ', '_')}")
    return " ".join(parts).strip()

df["text_aug"] = df.apply(build_text, axis=1)

# ==========================
# 4) Train/Test/Val Splits
# ==========================
X = df["text_aug"].tolist()
Y = df["tag_list"].tolist()

# Testsplit (Holdout)
X_train_raw, X_test_raw, y_train_list, y_test_list = train_test_split(
    X, Y, test_size=0.2, random_state=42
)
# aus dem Trainingsanteil noch eine kleine Validation (für Threshold-Tuning)
X_tr_raw, X_val_raw, y_tr_list, y_val_list = train_test_split(
    X_train_raw, y_train_list, test_size=0.1, random_state=42
)

# ==========================
# 5) Vektorisierung (Word + Char TF-IDF)
# ==========================
word_vec = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    lowercase=True,
    min_df=2
)
char_vec = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    lowercase=False,
    min_df=3
)
vectorizer = FeatureUnion([
    ("w", word_vec),
    ("c", char_vec),
])

X_tr_vec  = vectorizer.fit_transform(X_tr_raw)
X_val_vec = vectorizer.transform(X_val_raw)
X_test_vec= vectorizer.transform(X_test_raw)

# ==========================
# 6) MultiLabel Binarizer
# ==========================
mlb = MultiLabelBinarizer()
y_tr_bin  = mlb.fit_transform(y_tr_list)
y_val_bin = mlb.transform(y_val_list)
y_test_bin= mlb.transform(y_test_list)

# ==========================
# 7) Klassifikator (class_weight + etwas stärkeres C)
# ==========================
base = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    C=2.0,
    solver="liblinear"  # robust bei vielen Features; OneVsRest parallelisiert über Klassen
)
clf = OneVsRestClassifier(base, n_jobs=-1)
clf.fit(X_tr_vec, y_tr_bin)

# ==========================
# 8) Per-Class Threshold Tuning auf Validation
# ==========================
def tune_thresholds(clf, X_val_vec, y_val_bin, grid=None):
    if grid is None:
        grid = np.linspace(0.1, 0.9, 17)  # 0.1 .. 0.9 in 0.05-Schritten
    proba = clf.predict_proba(X_val_vec)
    n_classes = y_val_bin.shape[1]
    ths = np.full(n_classes, 0.5, dtype=float)
    for j in range(n_classes):
        best_th, best_f1 = 0.5, 0.0
        y_true = y_val_bin[:, j]
        if y_true.sum() == 0:
            # keine Positiven in Val für diese Klasse -> behalte 0.5
            continue
        for th in grid:
            y_pred = (proba[:, j] >= th).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1, best_th = f1, th
        ths[j] = best_th
    return ths

thresholds = tune_thresholds(clf, X_val_vec, y_val_bin)

# ==========================
# 9) Test-Evaluation mit Thresholds
# ==========================
proba_test = clf.predict_proba(X_test_vec)
y_pred_bin = (proba_test >= thresholds[None, :]).astype(int)

print("=== REPORT (micro/macro) ===")
print(classification_report(
    y_test_bin, y_pred_bin,
    target_names=mlb.classes_,
    zero_division=0
))

# ==========================
# 10) Artefakte speichern
# ==========================
os.makedirs("model_artifacts", exist_ok=True)
joblib.dump(vectorizer, "model_artifacts/tfidf_union.joblib")
joblib.dump(mlb,        "model_artifacts/labels.joblib")
joblib.dump(clf,        "model_artifacts/model.joblib")
joblib.dump(thresholds, "model_artifacts/thresholds.joblib")
print("Modelle gespeichert in ./model_artifacts")
