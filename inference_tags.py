import os
import numpy as np
import pandas as pd
import joblib
import sys
import io
import argparse

# ==========================
# 0) CONFIG
# ==========================
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--suffix",
    type=str,
    default="v7_c40",
    help="Versions-Suffix für alle Dokumente"
)

args = parser.parse_args()
SUFFIX = args.suffix

MODEL_DIR      = os.path.join(os.environ["PADDLE_DIR"], "model_artifacts")
PIPELINE_PATH  = os.path.join(MODEL_DIR, f"pipeline_{SUFFIX}.joblib")
LABELS_PATH    = os.path.join(MODEL_DIR, f"labels_{SUFFIX}.joblib")
THRESH_PATH    = os.path.join(MODEL_DIR, f"thresholds_{SUFFIX}.joblib")

sys.stdout.write(f"MODEL_DIR     = {MODEL_DIR}\n")
sys.stdout.write(f"PIPELINE_PATH = {PIPELINE_PATH}\n")
sys.stdout.write(f"LABELS_PATH   = {LABELS_PATH}\n")
sys.stdout.write(f"THRESH_PATH   = {THRESH_PATH}\n")
sys.stdout.flush()


# Spalten im CSV
# id,year,created,filename,title,author,text,tags
ID_COL       = "id"
YEAR_COL     = "year"
CREATED_COL  = "created"
FILENAME_COL = "filename"
TITLE_COL    = "title"
AUTHOR_COL   = "author"
TEXT_COL     = "text"
TAGS_COL     = "tags"  # optional, falls im Input schon vorhanden

TITLE_BOOST   = 2
AUTHOR_PREFIX = "__AUTHOR_"

# ==========================
# 1) MODELLE LADEN
# ==========================
print("[Info] Lade Modelle...")
PIPELINE   = joblib.load(PIPELINE_PATH)   # ColumnTransformer + ClassifierChain
MLB        = joblib.load(LABELS_PATH)

# Thresholds robust laden
raw_thresholds = joblib.load(THRESH_PATH)
THRESHOLDS     = np.asarray(raw_thresholds, dtype=float).ravel()

print(f"[Info] Anzahl Labels im Modell: {len(MLB.classes_)}")
print(f"[Info] Thresholds-Länge        : {THRESHOLDS.shape[0]}")

# ==========================
# 2) PREPROCESSING
# ==========================

def build_text(row: pd.Series) -> str:
    """
    Baut den gleichen zusammengesetzten Text wie beim Training:
    - Basis: text
    - Titel mehrfach angehängt (TITLE_BOOST)
    - Autor als stabiler Token AUTHOR_PREFIX + name
    """
    text   = str(row.get(TEXT_COL, "") or "")
    title  = str(row.get(TITLE_COL, "") or "").strip()
    author = str(row.get(AUTHOR_COL, "") or "").strip()

    parts = [text]

    if title:
        parts.append((" " + title) * TITLE_BOOST)

    if author:
        author_token = AUTHOR_PREFIX + author.lower().replace(" ", "_")
        parts.append(author_token)

    return " ".join(parts).strip()


def add_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Erzeugt die Spalten, die die Pipeline erwartet:
    - text_aug (TF-IDF)
    """
    df["text_aug"] = df.apply(build_text, axis=1)
    return df[["text_aug"]]


# ==========================
# 3) PREDICT-FUNKTIONEN
# ==========================

def predict_for_df(df: pd.DataFrame):
    """
    Nimmt ein DF mit Spalten (title,author,text,...) und gibt:
    - pred_tags_list (Liste von Listen mit Label-Strings)
    - proba (n_samples x n_labels)
    zurück.
    """
    X_feat = add_model_features(df.copy())

    proba = PIPELINE.predict_proba(X_feat)
    proba = np.asarray(proba)

    ths = THRESHOLDS
    n_classes = proba.shape[1]

    if ths.shape[0] != n_classes:
        print(f"[WARN] thresholds len {ths.shape[0]} != n_classes {n_classes} – passe an.")
        if ths.shape[0] > n_classes:
            ths = ths[:n_classes]
        else:
            pad = np.full(n_classes - ths.shape[0], 0.5, dtype=float)
            ths = np.concatenate([ths, pad])

    mask = proba >= ths[None, :]
    pred_indices_per_row = [np.where(mask[i])[0] for i in range(mask.shape[0])]

    labels = MLB.classes_
    pred_tags_list = [[labels[j] for j in inds] for inds in pred_indices_per_row]

    return pred_tags_list, proba


def predict_for_csv_path(csv_path: str):
    """
    - CSV von Pfad einlesen
    - Tags vorhersagen
    - Spalte 'predicted_tags' hinzufügen (kommagetrennt)
    - optional als neues CSV speichern
    """
    df = pd.read_csv(csv_path)

    # Falls Spalten fehlen, auffüllen
    for col in [ID_COL, YEAR_COL, CREATED_COL, FILENAME_COL, TITLE_COL, AUTHOR_COL, TEXT_COL]:
        if col not in df.columns:
            df[col] = ""

    pred_tags_list, proba = predict_for_df(df)

    df["predicted_tags"] = [", ".join(tags) for tags in pred_tags_list]

    return df, proba

# ==========================
# 4) MAIN-LOOP (für Java / CLI)
# ==========================
sys.stdout.write("inference ready\n")
sys.stdout.write("waiting\n")
sys.stdout.flush()

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    if line == "quit":
        break

    # Erwartetes Format:
    #  - "input.csv"
    df_out, _ = predict_for_csv_path(line)

    # Beispiel: Tags der ersten Zeile ausgeben (falls vorhanden)
    first_tags = ""
    if "predicted_tags" in df_out.columns and len(df_out) > 0:
        first_tags = df_out.loc[0, "predicted_tags"]

    sys.stdout.write(first_tags + "\n")
    sys.stdout.flush()

    sys.stdout.write("done\n")
    sys.stdout.flush()

    sys.stdout.write("waiting\n")
    sys.stdout.flush()
