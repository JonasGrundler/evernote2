import os
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
from sklearn.metrics import classification_report, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
import joblib

# ==========================
# 0) CONFIG
# ==========================
CSV_PATH    = r"C:\Users\Jonas\Downloads\EnTagging\csv\summary.csv"  # <- anpassen
TEXT_COL    = "text"
TAGS_COL    = "tags"
TITLE_COL   = "title"    # falls nicht vorhanden, wird leer gesetzt
AUTHOR_COL  = "author"   # falls nicht vorhanden, wird leer gesetzt
YEAR_COL    = "year"     # NEU: Jahres-Spalte (z. B. 2024)

MIN_FREQ    = 10         # Mindesthäufigkeit je Label
TITLE_BOOST = 2          # Titel 2x anhängen
AUTHOR_PREFIX = "__AUTHOR_"  # Autor als stabiler Token

# Labels mergen (Beispiel)
MERGE = {
    # "Whg 1 Karl-Pfaff-Straße": "Whg Karl-Pfaff-Straße",
    # "Whg 2 Karl-Pfaff-Straße": "Whg Karl-Pfaff-Straße",
    # "Whgn Karl-Pfaff-Straße":  "Whg Karl-Pfaff-Straße",
    # weitere Synonyme hier …
}

# ==========================
# 1) CSV einlesen & vorbereiten
# ==========================
df = pd.read_csv(CSV_PATH)

def split_tags(x):
    return [t.strip() for t in str(x).split(",") if t.strip()]

df["tag_list"] = df[TAGS_COL].apply(split_tags)

# ==========================
# 2) Label-Normalisierung & -Filter
# ==========================
def normalize_label(t):
    return MERGE.get(t, t)

df["tag_list"] = df["tag_list"].apply(lambda tags: [normalize_label(t) for t in tags])
cnt = Counter(t for tags in df["tag_list"] for t in tags)
df["tag_list"] = df["tag_list"].apply(lambda tags: [t for t in tags if cnt[t] >= MIN_FREQ])
df = df[df["tag_list"].map(len) > 0].reset_index(drop=True)

print(f"[Info] Datensätze nach Label-Filter: {len(df)}; unterschiedliche Labels: {len(set(t for tags in df['tag_list'] for t in tags))}")

# ==========================
# 3) Jahr & Text-Features bauen
# ==========================
# Jahr robust in Kategorie umwandeln (OHE)
year_num = pd.to_numeric(df[YEAR_COL], errors="coerce")
df["year_cat"] = np.where(year_num.notna(), year_num.astype(int).astype(str), "UNK")

def build_text(row):
    text   = str(row[TEXT_COL] or "")
    title  = str(row[TITLE_COL] or "").strip()
    author = str(row[AUTHOR_COL] or "").strip()

    parts = [text]
    if title:
        parts.append((" " + title) * TITLE_BOOST)
    if author:
        parts.append(f"{AUTHOR_PREFIX}{author.lower().replace(' ', '_')}")
    return " ".join(parts).strip()

df["text_aug"] = df.apply(build_text, axis=1)

# ==========================
# 4) Train/Test/Val Splits
# ==========================
X_df = df[["text_aug", "year_cat"]].copy()
Y = df["tag_list"].tolist()

X_train_df, X_test_df, y_train_list, y_test_list = train_test_split(
    X_df, Y, test_size=0.2, random_state=42
)
X_tr_df, X_val_df, y_tr_list, y_val_list = train_test_split(
    X_train_df, y_train_list, test_size=0.1, random_state=42
)

# ==========================
# 5) Vektorisierung (Word+Char TF-IDF) + Jahr (OneHot) & Klassifikator
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
text_union = FeatureUnion([
    ("w", word_vec),
    ("c", char_vec),
])

preprocess = ColumnTransformer(
    transformers=[
        ("txt", text_union, "text_aug"),
        ("year", OneHotEncoder(handle_unknown="ignore"), ["year_cat"]),
    ]
)

base = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    C=2.0,
    solver="liblinear"
)
ovr = OneVsRestClassifier(base, n_jobs=-1)

pipe = Pipeline(steps=[
    ("prep", preprocess),
    ("clf", ovr),
])

# MultiLabel-Binarizer
mlb = MultiLabelBinarizer()
y_tr_bin  = mlb.fit_transform(y_tr_list)
y_val_bin = mlb.transform(y_val_list)
y_test_bin= mlb.transform(y_test_list)

# Train
pipe.fit(X_tr_df, y_tr_bin)

# ==========================
# 5b) Per-Class Threshold Tuning (auf Validation)
# ==========================
def tune_thresholds(model, X_val, y_val_bin, grid=None):
    if grid is None:
        grid = np.linspace(0.1, 0.9, 17)
    proba = model.predict_proba(X_val)
    n_classes = y_val_bin.shape[1]
    ths = np.full(n_classes, 0.5, dtype=float)
    for j in range(n_classes):
        best_th, best_f1 = 0.5, 0.0
        y_true = y_val_bin[:, j]
        if y_true.sum() == 0:
            continue
        for th in grid:
            y_pred = (proba[:, j] >= th).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1, best_th = f1, th
        ths[j] = best_th
    return ths

thresholds = tune_thresholds(pipe, X_val_df, y_val_bin)

# ==========================
# 5c) Test-Evaluation (mit Thresholds)
# ==========================
proba_test = pipe.predict_proba(X_test_df)
y_pred_bin = (proba_test >= thresholds[None, :]).astype(int)

print("=== REPORT (micro/macro) ===")
print(classification_report(
    y_test_bin, y_pred_bin,
    target_names=mlb.classes_,
    zero_division=0
))

# ==========================
# Artefakte speichern
# ==========================
os.makedirs("model_artifacts", exist_ok=True)
joblib.dump(pipe,       "model_artifacts/pipeline.joblib")   # enthält Vektorisierung+OHE+Modell
joblib.dump(mlb,        "model_artifacts/labels.joblib")
joblib.dump(thresholds, "model_artifacts/thresholds.joblib")
print("Modelle gespeichert in ./model_artifacts")
