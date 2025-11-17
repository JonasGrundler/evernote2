import os
import re
import numpy as np
import pandas as pd
import argparse
from datetime import datetime
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import ClassifierChain  # <--- NEU
import joblib

# ==========================
# 0) CONFIG
# ==========================
print("start training " + datetime.now().strftime("%H:%M:%S"))
CSV_PATH      = os.getenv("CSV_PATH")  # r"C:\Users\Jonas\.jg-evernote\enex-batch\csv\summary.csv"
INT_PATH      = os.getenv("INT_PATH")
TEXT_COL      = "text"
TAGS_COL      = "tags"
TITLE_COL     = "title"    # falls nicht vorhanden, wird leer gesetzt
AUTHOR_COL    = "author"   # falls nicht vorhanden, wird leer gesetzt

MIN_FREQ      = 10         # Mindesthäufigkeit je Label
TITLE_BOOST   = 1          # Titel 2x anhängen
AUTHOR_PREFIX = "__AUTHOR_"  # Autor als stabiler Token

# Labels mergen (Beispiel)
MERGE = {
    # "Whg 1 Karl-Pfaff-Straße": "Whg Karl-Pfaff-Straße",
    # "Whg 2 Karl-Pfaff-Straße": "Whg Karl-Pfaff-Straße",
    # "Whgn Karl-Pfaff-Straße":  "Whg Karl-Pfaff-Straße",
}

# Für ClassifierChain: Labels, deren Info für andere wichtig ist (z.B. Jahr)
# Diese Labels werden in der Kette nach vorne gezogen, damit andere Labels
# deren Vorhersage "sehen" können.
PRIORITY_TAGS = [
    "Steuer",
    "SDK",
    "Whg 1 Karl-Pfaff-Straße", "Whg 2 Karl-Pfaff-Straße",
    "Whg Sonnenberg", "Whg Neue Weinsteige",
    "Whg Leuschnerstraße", "Whg Schwabstraße",
    "Whg Singen", "Whg Schwieberdingen",
    "Whg Degerloch", "verdimo"
]   # kannst du beliebig erweitern

EXCLUDE_LABELS = {"done"}

parser = argparse.ArgumentParser()
parser.add_argument(
    "--suffix",
    type=str,
    default="v5",
    help="Versions-Suffix für die Artefakt-Dateinamen, z.B. 'v5', 'v6_experiment1'"
)
parser.add_argument(
    "--min_year",
    type=int,
    default=1900,
    help="Nur Zeilen verwenden, deren Jahr >= diesem Jahr ist (z.B. 2020)"
)
parser.add_argument(
    "--max_year",
    type=int,
    default=2300,
    help="Nur Zeilen verwenden, deren Jahr <= diesem Jahr ist (z.B. 2020)"
)
args = parser.parse_args()
SUFFIX = args.suffix
MIN_YEAR = args.min_year
MAX_YEAR = args.max_year


def is_excluded_label(t: str) -> bool:
    """
    True für alle Labels, die wir NICHT vom Modell vorhersagen lassen wollen.
    - Jahreszahlen
    - bestimmte Tags wie 'done'
    """
    t = str(t).strip()

    # je nach Taste kannst du hier case-sensitiv oder -insensitiv arbeiten:
    if t.lower() in {lbl.lower() for lbl in EXCLUDE_LABELS}:
        return True
    return False

def is_year_label(t: str) -> bool:
    """
    True, wenn Label eine reine 4-stellige Jahreszahl ist (z.B. '2023').
    """
    t = str(t).strip()
    if not re.fullmatch(r"\d{4}", t):
        return False
    year = int(t)
    return 1000 <= year <= 3000

# ==========================
# 1) CSV einlesen & vorbereiten
# ==========================
df1 = pd.read_csv(CSV_PATH)

if INT_PATH is not None:
    if Path(INT_PATH).exists():
        df2 = pd.read_csv(INT_PATH)
        df = pd.concat([df1, df2], ignore_index=True)
    else:
        df = df1
else:
    df = df1

df = df[df["year"] >= MIN_YEAR].copy()

def split_tags(x):
    return [t.strip() for t in str(x).split(",") if t.strip()]

df["tag_list"] = df[TAGS_COL].apply(split_tags)

# ==========================
# 2) Label-Normalisierung & -Filter
# ==========================
def normalize_label(t):
    return MERGE.get(t, t)

df["tag_list"] = df["tag_list"].apply(
    lambda tags: [normalize_label(t) for t in tags]
)

# Jahres-Labels entfernen (z.B. '2020', '2021', '2022', ...)
df["tag_list"] = df["tag_list"].apply(
    lambda tags: [t for t in tags if not is_year_label(t)]
)

# EXCLUDE_LABELS entfernen
df["tag_list"] = df["tag_list"].apply(
    lambda tags: [t for t in tags if not is_excluded_label(t)]
)

# Label-Frequenzen
cnt = Counter(t for tags in df["tag_list"] for t in tags)

# seltene Labels entfernen
df["tag_list"] = df["tag_list"].apply(
    lambda tags: [t for t in tags if cnt[t] >= MIN_FREQ]
)
df = df[df["tag_list"].map(len) > 0].reset_index(drop=True)

print(
    f"[Info] Datensätze nach Label-Filter: {len(df)}; "
    f"unterschiedliche Labels: {len(set(t for tags in df['tag_list'] for t in tags))}"
)

# ==========================
# 3) Text-Features bauen
# ==========================

def build_text(row):
    text   = str(row.get(TEXT_COL, "") or "")
    title  = str(row.get(TITLE_COL, "") or "").strip()
    author = str(row.get(AUTHOR_COL, "") or "").strip()

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
X_df = df[["text_aug"]].copy()
Y = df["tag_list"].tolist()

X_train_df, X_test_df, y_train_list, y_test_list = train_test_split(
    X_df, Y, test_size=0.2, random_state=42
)
X_tr_df, X_val_df, y_tr_list, y_val_list = train_test_split(
    X_train_df, y_train_list, test_size=0.1, random_state=42
)

# ==========================
# 5) Vektorisierung (Word+Char TF-IDF) + Jahr (OneHot)
# ==========================
word_vec_1 = TfidfVectorizer(
#    max_features=100000,
    ngram_range=(1, 1),
    lowercase=True,
    min_df=10
)
word_vec_2 = TfidfVectorizer(
#    max_features=100000,
    ngram_range=(2, 3),
    lowercase=True,
    min_df=8
)
char_vec = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    lowercase=True,
    min_df=5
)

text_union = FeatureUnion([
    ("w1", word_vec_1),
    ("w2", word_vec_2),
    ("c", char_vec),
])

preprocess = ColumnTransformer(
    transformers=[
        ("txt", text_union, "text_aug"),
    ]
)

# ==========================
# 5a) MultiLabel-Binarizer
# ==========================
mlb = MultiLabelBinarizer()
y_tr_bin   = mlb.fit_transform(y_tr_list)

y_val_bin  = mlb.transform(y_val_list)
y_test_bin = mlb.transform(y_test_list)

n_classes = y_tr_bin.shape[1]
classes   = list(mlb.classes_)
label_to_idx = {lbl: i for i, lbl in enumerate(classes)}

# ==========================
# 5b) Reihenfolge für ClassifierChain
# ==========================

# Prioritätslabels (z.B. "Steuer", "SDK") nach vorne
priority_indices = [label_to_idx[l] for l in PRIORITY_TAGS if l in label_to_idx]

# Restliche Labels
remaining_indices = [
    i for i in range(n_classes)
    if i not in priority_indices
]

# Endgültige Reihenfolge:
#   1. Prioritätslabels
#   2. Rest
order = priority_indices + remaining_indices

# ==========================
# 5c) ClassifierChain + Pipeline
# ==========================
base = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    C=4.0,
    solver="liblinear"
)

chain = ClassifierChain(
    estimator=base,
    order=order,
    random_state=42
)

pipe = Pipeline(steps=[
    ("prep", preprocess),
    ("clf", chain),
])

# ==========================
# 5d) Train
# ==========================
pipe.fit(X_tr_df, y_tr_bin)

# ==========================
# 5d-1) Anzahl TF-IDF-Features inspizieren
# ==========================
# Den gefitteten ColumnTransformer holen
prep_fitted = pipe.named_steps["prep"]

# Daraus den FeatureUnion-Block "txt"
union_fitted = prep_fitted.named_transformers_["txt"]

# Aus der FeatureUnion die drei Vektorisierer holen
w1_fitted = union_fitted.transformer_list[0][1]  # ("w1", word_vec_1)
w2_fitted = union_fitted.transformer_list[1][1]  # ("w2", word_vec_2)
c_fitted  = union_fitted.transformer_list[2][1]  # ("c",  char_vec)

print("Word uni-gram Features (w1):", len(w1_fitted.get_feature_names_out()))
print("Word 2–3-gram Features (w2):", len(w2_fitted.get_feature_names_out()))
print("Char 3–5-gram Features (c):", len(c_fitted.get_feature_names_out()))
print("Gesamt-Textfeatures (ungefähr):",
      len(w1_fitted.get_feature_names_out())
    + len(w2_fitted.get_feature_names_out())
    + len(c_fitted.get_feature_names_out()))

# ==========================
# 5e) Helper: proba-Matrix aus OVR oder ClassifierChain
# ==========================
def get_proba_matrix(model, X):
    """
    Vereinheitlicht die Ausgabe von predict_proba:
    - OneVsRestClassifier: 2D (n_samples, n_classes)
    - ClassifierChain: 1D Array, jedes Element (n_classes,)
    """
    proba_raw = model.predict_proba(X)
    if isinstance(proba_raw, np.ndarray) and proba_raw.ndim == 1:
        # z.B. ClassifierChain: Array mit Shape (n_samples,), jedes Element (n_classes,)
        return np.vstack(proba_raw)
    return proba_raw  # OneVsRest etc.


# ==========================
# 5f) Per-Class Threshold Tuning (auf Validation)
# ==========================
def tune_thresholds(model, X_val, y_val_bin, grid=None):
    if grid is None:
        grid = np.linspace(0.1, 0.9, 17)

    proba = get_proba_matrix(model, X_val)
    n_classes = y_val_bin.shape[1]
    ths = np.full(n_classes, 0.5, dtype=float)

    for j in range(n_classes):
        y_true = y_val_bin[:, j]
        if y_true.sum() == 0:
            continue

        best_th, best_f1 = 0.5, 0.0
        for th in grid:
            y_pred = (proba[:, j] >= th).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1, best_th = f1, th
        ths[j] = best_th

    return ths

thresholds = tune_thresholds(pipe, X_val_df, y_val_bin)

# ==========================
# 5g) Test-Evaluation (mit Thresholds)
# ==========================
proba_test = get_proba_matrix(pipe, X_test_df)
y_pred_bin = (proba_test >= thresholds[None, :]).astype(int)

print("=== REPORT (micro/macro) ===")
report_str = classification_report(
    y_test_bin,
    y_pred_bin,
    target_names=mlb.classes_,
    zero_division=0
)

print(report_str)

# ==========================
# Artefakte speichern
# ==========================

out_dir = f"model_artifacts"

os.makedirs("model_artifacts", exist_ok=True)
joblib.dump(pipe,       os.path.join(out_dir, f"pipeline_{SUFFIX}.joblib"))
joblib.dump(mlb,        os.path.join(out_dir, f"labels_{SUFFIX}.joblib"))
joblib.dump(thresholds, os.path.join(out_dir, f"thresholds_{SUFFIX}.joblib"))

#joblib.dump(pipe,       "model_artifacts/pipeline_v5.joblib")  # cc = ClassifierChain
#joblib.dump(mlb,        "model_artifacts/labels_v5.joblib")
#joblib.dump(thresholds, "model_artifacts/thresholds_v5.joblib")

report_path = os.path.join(out_dir, f"report_{SUFFIX}.txt")

with open(report_path, "w", encoding="utf-8") as f:
    f.write("=== REPORT (micro/macro) ===\n")
    f.write(report_str)

print("ended training " + datetime.now().strftime("%H:%M:%S"))
