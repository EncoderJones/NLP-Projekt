import gzip, json, random
import pandas as pd
from pathlib import Path

random.seed(42)

BASE = Path(__file__).resolve().parent.parent
RAW  = BASE / "data" / "raw"
PROC = BASE / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)

FILES = {
    "Books":        RAW / "books.jsonl.gz",
    "Electronics":  RAW / "electronics.jsonl.gz",
    "Video Games":  RAW / "videogames.json.gz",
}

def sample_category(path, category, n=5000, max_scan=300000):
    """liest bis zu max_scan Zeilen, zieht n gültige Reviews"""
    buf = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_scan:
                break
            try:
                raw = json.loads(line)
                rating = raw.get("rating")
                title  = raw.get("title")
                text   = raw.get("text")
                if text and rating in [1.0,2.0,3.0,4.0,5.0]:
                    buf.append({"title": title, "text": text, "stars": int(rating)})
            except:
                continue
    if len(buf) < n:
        raise RuntimeError(f"Zu wenige gültige Reviews in {category}. Gefunden: {len(buf)}")
    sample = random.sample(buf, n)
    df = pd.DataFrame(sample)
    df["category"] = category
    return df

# pro Kategorie verarbeiten
all_frames = []
for cat, path in FILES.items():
    df = sample_category(path, cat, n=5000)
    out_single = PROC / f"{cat.replace(' ', '_').lower()}_sample.csv"
    df.to_csv(out_single, index=False)
    print(f"Gespeichert: {out_single}")
    all_frames.append(df)

full = pd.concat(all_frames, ignore_index=True)
full_out = PROC / "reviews_sample_3cats.csv"
full.to_csv(full_out, index=False)
print(f"Gesamtdatei gespeichert: {full_out}")
print("Anzahl Zeilen gesamt:", len(full))
