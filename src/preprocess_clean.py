import re
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
PROC = BASE / "data" / "processed"

inp = PROC / "reviews_sample_3cats_split.csv"   
out = PROC / "reviews_clean.csv"

URL_RE   = re.compile(r"http[s]?://\S+|www\.\S+")
HTML_RE  = re.compile(r"<[^>]+>")
NONALPH  = re.compile(r"[^a-z\s]")

def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower()                         
    s = URL_RE.sub(" ", s)                 
    s = HTML_RE.sub(" ", s)             
    s = NONALPH.sub(" ", s)                
    s = re.sub(r"\s+", " ", s).strip()   
    return s

def main():
    df = pd.read_csv(inp)
    before = len(df)

    df["text_clean"] = df["text"].apply(clean_text)

    # Sehr kurze Texte filtern
    wc = df["text_clean"].str.split().str.len().fillna(0)
    df = df[wc >= 5].reset_index(drop=True)
    after = len(df)

    df.to_csv(out, index=False)

    print("Gespeichert:", out)
    print("Anzahl Zeilen vor/nach:", before, "/", after)
    print("Spalten:", list(df.columns))
    print("\nVorher/Nachher-Beispiel:")
    raw_sample = pd.read_csv(inp, nrows=1)["text"].iloc[0]
    print("raw :", str(raw_sample)[:200].replace("\n", " "))
    print("clean:", clean_text(str(raw_sample))[:200])

if __name__ == "__main__":
    main()