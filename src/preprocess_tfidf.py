from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import scipy.sparse as sp

BASE = Path(__file__).resolve().parent.parent
PROC = BASE / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)

inp = PROC / "reviews_lemma.csv"

df = pd.read_csv(inp)

# Eingabe: Titel + Text_lem zusammenführen
def merge_title_text(row):
    title = str(row.get("title", "")) if isinstance(row.get("title", ""), str) else ""
    text  = str(row.get("text_lem", ""))
    return (title + " " + text).strip()

df["text_in"] = df.apply(merge_title_text, axis=1)
y = df["stars"].astype(int).values
split = df["split"]

# Train/Test-Sets nach Split-Spalte
X_train_text = df.loc[split == "train", "text_in"].tolist()
y_train      = y[split == "train"]
X_test_text  = df.loc[split == "test", "text_in"].tolist()
y_test       = y[split == "test"]

# TF-IDF: unigrams+bigrams
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),     # Unigramme + Bigramme
    min_df=5,               
    max_df=0.90,           
    sublinear_tf=True,      
    max_features=30000,     # Deckelung auf die 30.000 wichtigsten Features
    lowercase=False         
)


X_train = vectorizer.fit_transform(X_train_text)
X_test  = vectorizer.transform(X_test_text)

joblib.dump(vectorizer, PROC / "tfidf_vectorizer.joblib")
sp.save_npz(PROC / "X_train_tfidf.npz", X_train)
sp.save_npz(PROC / "X_test_tfidf.npz",  X_test)
joblib.dump(y_train, PROC / "y_train.joblib")
joblib.dump(y_test,  PROC / "y_test.joblib")

print("Gespeichert:")
print(" -", PROC / "tfidf_vectorizer.joblib")
print(" -", PROC / "X_train_tfidf.npz", X_train.shape)
print(" -", PROC / "X_test_tfidf.npz",  X_test.shape)
print(" -", PROC / "y_train.joblib", len(y_train))
print(" -", PROC / "y_test.joblib",  len(y_test))