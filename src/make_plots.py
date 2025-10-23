import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent
PROC = BASE / "data" / "processed"
FIGS = PROC / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(PROC / "reviews_sample_3cats_split.csv")

#  Plot 1: Gesamtverteilung Sterne
star_total = df["stars"].value_counts().sort_index()

plt.figure(figsize=(6,4))
plt.bar(star_total.index.astype(str), star_total.values)
plt.title("Gesamtverteilung der Sternebewertungen")
plt.xlabel("Sterne")
plt.ylabel("Anzahl")
plt.tight_layout()
plt.savefig(FIGS / "plot_star_total.png")  
plt.close()

# Plot 2: Sterne pro Kategorie 
star_by_cat = df.groupby(["category","stars"]).size().unstack(fill_value=0).reindex(columns=[1,2,3,4,5])

cats = star_by_cat.index.tolist()
stars = star_by_cat.columns.tolist()
x = range(len(cats))
width = 0.15

plt.figure(figsize=(8,4))
for i, s in enumerate(stars):
    plt.bar([xi + (i-2)*width for xi in x], star_by_cat[s].values, width, label=str(s))
plt.xticks([xi for xi in x], cats)
plt.title("Sterneverteilung pro Kategorie")
plt.xlabel("Kategorie")
plt.ylabel("Anzahl")
plt.legend(title="Sterne", ncol=5, fontsize=8, title_fontsize=9)
plt.tight_layout()
plt.savefig(FIGS / "plot_star_by_category.png")
plt.close()

# Plot 3: Train/Test je Kategorie
split_counts = df.groupby(["category","split"]).size().unstack(fill_value=0)[["train","test"]]

x = range(len(split_counts.index))
width = 0.35

plt.figure(figsize=(6,4))
plt.bar([xi - width/2 for xi in x], split_counts["train"].values, width, label="Train")
plt.bar([xi + width/2 for xi in x], split_counts["test"].values,  width, label="Test")
plt.xticks([xi for xi in x], split_counts.index)
plt.title("Train/Test-Aufteilung je Kategorie (80/20)")
plt.xlabel("Kategorie")
plt.ylabel("Anzahl")
plt.legend()
plt.tight_layout()
plt.savefig(FIGS / "plot_train_test_by_category.png")
plt.close()

print("Fertig. PNG-Dateien gespeichert in:", FIGS)