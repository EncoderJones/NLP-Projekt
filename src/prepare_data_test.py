import gzip
import json
import random
import pandas as pd
from pathlib import Path

random.seed(42)

# Pfad zur Books-Datei
path = Path(__file__).resolve().parent.parent / "data" / "raw" / "books.jsonl.gz"

reviews = []

with gzip.open(path, "rt", encoding="utf-8") as f:
    for i, line in enumerate(f):
        try:
            raw = json.loads(line)
            rating = raw.get("rating")
            title = raw.get("title")
            text = raw.get("text")
            if rating in [1.0, 2.0, 3.0, 4.0, 5.0] and text:
                reviews.append({"title": title, "text": text, "stars": int(rating)})
        except Exception:
            continue
        if i > 200000:  
            break

# Zufällig 5000 auswählen
sample = random.sample(reviews, 5000)

# In DataFrame umwandeln
df = pd.DataFrame(sample)
df["category"] = "Books"

# In processed speichern
outpath = Path(__file__).resolve().parent.parent / "data" / "processed" / "books_sample_test.csv"
df.to_csv(outpath, index=False)

print("Fertig! Books-Sample gespeichert:", outpath)
print(df.head())
print("Gesamtanzahl:", len(df))
print(df.columns)
print(df[["stars", "category"]].head())
