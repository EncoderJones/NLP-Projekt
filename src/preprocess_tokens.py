import pandas as pd
from pathlib import Path
import spacy

BASE = Path(__file__).resolve().parent.parent
PROC = BASE / "data" / "processed"

inp = PROC / "reviews_clean.csv"      
out = PROC / "reviews_tokens.csv"     

nlp = spacy.load("en_core_web_sm", exclude=["ner"])

def tokenize_and_filter(text):
    if not isinstance(text, str):
        return []
    doc = nlp(text)
    tokens = [tok.text.lower() for tok in doc if tok.is_alpha and not tok.is_stop]
    return tokens

def main():
    df = pd.read_csv(inp)
    print("Starte Tokenisierung für", len(df), "Texte...")

    df["tokens"] = df["text_clean"].apply(tokenize_and_filter)

    print(df[["text_clean", "tokens"]].head(3))

    df.to_csv(out, index=False)
    print("Gespeichert:", out)

if __name__ == "__main__":
    main()