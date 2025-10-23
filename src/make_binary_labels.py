from pathlib import Path
import joblib
import numpy as np

BASE = Path(__file__).resolve().parent.parent
PROC = BASE / "data" / "processed"

y_train = joblib.load(PROC / "y_train.joblib")
y_test  = joblib.load(PROC / "y_test.joblib")

def to_sentiment(y_stars):
    # 1–2 -> 0 (negativ), 3 -> 1 (neutral), 4–5 -> 2 (positiv)
    y_stars = np.asarray(y_stars, dtype=int)
    y_bin = np.empty_like(y_stars)
    y_bin[y_stars <= 2] = 0
    y_bin[y_stars == 3] = 1
    y_bin[y_stars >= 4] = 2
    return y_bin

y_train_bin = to_sentiment(y_train)
y_test_bin  = to_sentiment(y_test)

joblib.dump(y_train_bin, PROC / "y_train_bin.joblib")
joblib.dump(y_test_bin,  PROC / "y_test_bin.joblib")

print("Gespeichert:")
print(" -", PROC / "y_train_bin.joblib", len(y_train_bin))
print(" -", PROC / "y_test_bin.joblib",  len(y_test_bin))
print("Klassen-Codierung: 0=negativ, 1=neutral, 2=positiv")