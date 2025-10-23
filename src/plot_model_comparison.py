from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parent.parent
PROC = BASE / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)

five = {
    "Naive Bayes":         {"Accuracy": 0.636, "Macro-F1": 0.228},
    "Logistic Regression": {"Accuracy": 0.691, "Macro-F1": 0.500},
    "Neuronales Netz":     {"Accuracy": 0.698, "Macro-F1": 0.444},
}

binary = {
    "Naive Bayes":         {"Accuracy": 0.8045, "Macro-F1": 0.3838},
    "Logistic Regression": {"Accuracy": 0.8625, "Macro-F1": 0.6603},
    "Neuronales Netz":     {"Accuracy": 0.8610, "Macro-F1": 0.6275},
}

def plot_task(ax, data, title):
    models = list(data.keys())
    acc = [data[m]["Accuracy"] for m in models]
    f1  = [data[m]["Macro-F1"] for m in models]

    x = np.arange(len(models))
    w = 0.35

    b1 = ax.bar(x - w/2, acc, width=w, label="Accuracy")
    b2 = ax.bar(x + w/2, f1,  width=w, label="Macro-F1")

    for bars in (b1, b2):
        for rect in bars:
            h = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2, h + 0.01, f"{h:.3f}",
                    ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=0)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Wert")
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(loc="lower right")

def main():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    plot_task(axes[0], five,   "Modellvergleich – 5 Klassen (1–5 Sterne)")
    plot_task(axes[1], binary, "Modellvergleich – Binary Sentiment (neg/neu/pos)")

    out = PROC / "modelvergleich_metrics.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print("Diagramm gespeichert:", out)

if __name__ == "__main__":
    main()