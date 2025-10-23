import pandas as pd
from pathlib import Path
import random

random.seed(42)

BASE = Path(__file__).resolve().parent.parent
PROC = BASE / "data" / "processed"

df = pd.read_csv(PROC / "reviews_sample_3cats.csv")

# Gesamtverteilung Sterne
star_total = df["stars"].value_counts().sort_index()
print("\nTabelle X – Gesamtverteilung Sterne:")
print(star_total)

# Verteilung Sterne pro Kategorie
star_by_cat = df.groupby(["category", "stars"]).size().unstack(fill_value=0)
print("\nTabelle Y – Verteilung Sterne pro Kategorie:")
print(star_by_cat)

def stratified_split_exact(data, test_frac=0.2, seed=42):
    rng = random.Random(seed)
    parts = []
    for cat, cat_df in data.groupby("category", sort=False):
        cat_df = cat_df.copy()  
        n_total = len(cat_df)
        n_test_target = round(n_total * test_frac)

        alloc = {star: round(len(grp) * test_frac) for star, grp in cat_df.groupby("stars")}

        diff = n_test_target - sum(alloc.values())
        if diff != 0:
            sizes = sorted([(len(cat_df[cat_df["stars"] == s]), s) for s in alloc], reverse=True)
            idx = 0
            step = 1 if diff > 0 else -1
            while diff != 0:
                s = sizes[idx % len(sizes)][1]
                alloc[s] += step
                diff -= step
                idx += 1

        cat_df["split"] = "train"
        for star, grp in cat_df.groupby("stars"):
            k = max(0, min(len(grp), alloc[star]))
            test_idx = rng.sample(list(grp.index), k)
            cat_df.loc[test_idx, "split"] = "test"

        parts.append(cat_df)

    return pd.concat(parts, ignore_index=True)

df_split = stratified_split_exact(df, test_frac=0.2, seed=42)
split_counts = df_split.groupby(["category", "split"]).size().unstack(fill_value=0)
print("\nTabelle Z – Train/Test-Split (exakt 80/20):")
print(split_counts)

star_total.to_csv(PROC / "stats_star_total.csv", header=["count"])
star_by_cat.to_csv(PROC / "stats_star_by_category.csv")
split_counts.to_csv(PROC / "stats_train_test.csv")
df_split.to_csv(PROC / "reviews_sample_3cats_split.csv", index=False)

from tabulate import tabulate

print("\nTabelle X (Gesamtverteilung):")
print(tabulate(star_total.reset_index().values, headers=["Sterne", "Anzahl"], tablefmt="github"))

print("\nTabelle Y (pro Kategorie):")
print(tabulate(star_by_cat.reset_index().values, headers=["Kategorie","1 Stern","2 Sterne","3 Sterne","4 Sterne","5 Sterne"], tablefmt="github"))

print("\nTabelle Z (Train/Test-Split):")
print(tabulate(split_counts.reset_index().values, headers=["Kategorie","Test","Train"], tablefmt="github"))
