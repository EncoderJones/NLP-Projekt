from pathlib import Path
import joblib
from scipy.sparse import load_npz
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

BASE = Path(__file__).resolve().parent.parent
PROC = BASE / "data" / "processed"

X_test = load_npz(PROC / "X_test_tfidf.npz")
y_test = joblib.load(PROC / "y_test.joblib")

def to_binary_labels(y_stars):
    y_stars = np.asarray(y_stars)
    y_bin = np.full_like(y_stars, fill_value=2)  
    y_bin[(y_stars == 3)] = 1                   
    y_bin[(y_stars <= 2)] = 0                   
    return y_bin

y_test_bin = to_binary_labels(y_test)


models = []
for name, fname in [
    ("Naive Bayes", "model_nb.joblib"),
    ("Logistic Regression", "model_logreg.joblib"),
    ("Neuronales Netz", "model_nn.joblib"),
]:
    path = PROC / fname
    if path.exists():
        models.append((name, joblib.load(path)))
    else:
        print(f"[WARN] Modell nicht gefunden und wird übersprungen: {path}")

if not models:
    raise SystemExit("Keine Modelle gefunden. Bitte zuerst train_nb.py, train_logreg.py, train_nn.py ausführen.")

def compute_metrics(y_true, y_pred, avg="macro"):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average=avg, zero_division=0),
        "Recall": recall_score(y_true, y_pred, average=avg, zero_division=0),
        "F1": f1_score(y_true, y_pred, average=avg, zero_division=0),
    }

results_multi = {}   # 5-Klassen
results_bin   = {}   # 3-Klassen 

for name, clf in models:
    y_pred_multi = clf.predict(X_test)
    metrics_multi = compute_metrics(y_test, y_pred_multi, avg="macro")
    results_multi[name] = metrics_multi

    # Auf 3 Klassen mappen
    y_pred_bin = to_binary_labels(y_pred_multi)
    metrics_bin = compute_metrics(y_test_bin, y_pred_bin, avg="macro")
    results_bin[name] = metrics_bin

# Plot-Funktion
def plot_metrics_bar(results_dict, title, out_path):
    model_names = list(results_dict.keys())
    metric_names = ["Accuracy", "Precision", "Recall", "F1"]
 
    vals = np.array([[results_dict[m][k] for k in metric_names] for m in model_names])

    # Balkendiagramm
    fig, ax = plt.subplots(figsize=(8, 4.8))
    n_models, n_metrics = vals.shape
    index = np.arange(n_models)
    width = 0.18

    for j, metric in enumerate(metric_names):
        ax.bar(index + j*width, vals[:, j], width, label=metric)

    ax.set_ylabel("Wert")
    ax.set_ylim(0, 1.0)
    ax.set_xticks(index + width * (n_metrics-1) / 2)
    ax.set_xticklabels(model_names, rotation=0)
    ax.set_title(title)
    ax.legend(loc="lower right", ncol=2)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    fig.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Gespeichert: {out_path}")

plot_metrics_bar(results_multi,
                 "Modellvergleich (5 Klassen) – Accuracy / Precision / Recall / F1 (Macro)",
                 PROC / "metrics_multiclass.png")

plot_metrics_bar(results_bin,
                 "Modellvergleich (Binary-Sentiment) – Accuracy / Precision / Recall / F1 (Macro)",
                 PROC / "metrics_binary.png")