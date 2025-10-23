import pandas as pd
from pathlib import Path
import spacy

BASE = Path(__file__).resolve().parent.parent
PROC = BASE / "data" / "processed"

inp = PROC / "reviews_clean.csv"     
out = PROC / "reviews_lemma.csv"     

def build_nlp():
    nlp = spacy.load("en_core_web_sm", exclude=["ner"])
    return nlp

def lemmatize_batch(texts, nlp):
    results = []
    for doc in nlp.pipe(texts, batch_size=1000, n_process=1):
        lemmas = [
            tok.lemma_.lower()
            for tok in doc
            if tok.is_alpha and not tok.is_stop and len(tok) > 2
        ]
        results.append(" ".join(lemmas))
    return results

def main():
    df = pd.read_csv(inp)
    if "text_clean" not in df.columns:
        raise RuntimeError("Spalte 'text_clean' fehlt. Bitte zuerst preprocess_clean.py ausführen.")

    nlp = build_nlp()
    print("Lemmatisierung startet, Anzahl Texte:", len(df))

    texts = df["text_clean"].fillna("")
    df["text_lem"] = lemmatize_batch(texts, nlp)

    nonempty_before = (df["text_clean"].str.len() > 0).sum()
    nonempty_after  = (df["text_lem"].str.len() > 0).sum()
    print("Nicht-leere Texte vor/nach:", nonempty_before, "/", nonempty_after)

    df.to_csv(out, index=False)
    print("Gespeichert:", out)
    print(df[["text_clean", "text_lem"]].head(3))

if __name__ == "__main__":
    main()