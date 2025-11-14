import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

# ==========================
# 1) CSV EINLESEN
# ==========================
CSV_PATH = r"C:\Users\Jonas\Downloads\EnTagging\csv\summary.csv"  # <- anpassen

df = pd.read_csv(CSV_PATH)

# Spaltennamen hier zentral festlegen
TEXT_COL = "text"
TAGS_COL = "tags"
TITLE_COL = "title"     # <-- anpassen, falls anders
AUTHOR_COL = "author"   # <-- anpassen, z.B. "from" oder "sender"

# fehlende Spalten abfangen
if TITLE_COL not in df.columns:
    df[TITLE_COL] = ""
if AUTHOR_COL not in df.columns:
    df[AUTHOR_COL] = ""

# alles zu String machen
df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str)
df[TITLE_COL] = df[TITLE_COL].fillna("").astype(str)
df[AUTHOR_COL] = df[AUTHOR_COL].fillna("").astype(str)
df[TAGS_COL] = df[TAGS_COL].fillna("").astype(str)

# Tags in Liste umwandeln
def split_tags(x):
    return [t.strip() for t in str(x).split(",") if t.strip()]

df["tag_list"] = df[TAGS_COL].apply(split_tags)

# ==========================
# 1b) FEATURES BAUEN
# ==========================
# Titel und Autor extra „lauter“ machen
df["combined_text"] = (
    (df[TITLE_COL] + " ") * 3    # Titel 3x so wichtig
    + (df[AUTHOR_COL] + " ") * 2 # Autor 2x so wichtig
    + df[TEXT_COL]               # eigentlicher Inhalt
)

texts = df["combined_text"].tolist()
tags = df["tag_list"].tolist()

# ==========================
# 2) TRAIN / TEST SPLIT
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    texts, tags, test_size=0.2, random_state=42
)

# ==========================
# 3) TF-IDF
# ==========================
vectorizer = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    lowercase=True
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ==========================
# 4) TAGS BINARISIEREN
# ==========================
mlb = MultiLabelBinarizer()
y_train_bin = mlb.fit_transform(y_train)
y_test_bin = mlb.transform(y_test)

# ==========================
# 5) KLASSIFIKATOR
# ==========================
clf = OneVsRestClassifier(
    LogisticRegression(max_iter=400)
)
clf.fit(X_train_vec, y_train_bin)

# ==========================
# 6) EVALUATION
# ==========================
y_pred_bin = clf.predict(X_test_vec)

print("=== REPORT (micro/macro) ===")
target_names = mlb.classes_
print(classification_report(
    y_test_bin,
    y_pred_bin,
    target_names=target_names,
    zero_division=0
))

# ==========================
# 7) MODELLE SPEICHERN
# ==========================
os.makedirs("model_artifacts", exist_ok=True)
joblib.dump(vectorizer, "model_artifacts/tfidf.joblib")
joblib.dump(mlb, "model_artifacts/labels.joblib")
joblib.dump(clf, "model_artifacts/model.joblib")

print("Modelle gespeichert in ./model_artifacts")
