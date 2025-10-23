from pathlib import Path
from scipy.sparse import load_npz
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

BASE = Path(__file__).resolve().parent.parent
PROC = BASE / "data" / "processed"

X_train = load_npz(PROC / "X_train_tfidf.npz")
X_test  = load_npz(PROC / "X_test_tfidf.npz")
y_train = joblib.load(PROC / "y_train.joblib")
y_test  = joblib.load(PROC / "y_test.joblib")

clf = MLPClassifier(
    hidden_layer_sizes=(128,),   # Schicht mit 128 Neuronen
    activation="relu",
    solver="adam",
    alpha=1e-4,                  # L2-Regularisierung
    learning_rate_init=1e-3,     
    max_iter=200,               
    early_stopping=True,         # stoppt automatisch, wenn es stagniert
    validation_fraction=0.1,     # 10 % vom Training für Validierung
    n_iter_no_change=10,         
    random_state=42,
    verbose=True
)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average="macro")
cm  = confusion_matrix(y_test, y_pred)
rep = classification_report(y_test, y_pred, digits=3)

print("Accuracy:", round(acc, 4))
print("Macro-F1:", round(macro_f1, 4))
print("Confusion:\n", cm)
print("Report:\n", rep)

joblib.dump(clf, PROC / "model_nn.joblib")
with open(PROC / "results_nn.txt", "w") as f:
    f.write(f"Accuracy: {acc:.4f}\n")
    f.write(f"Macro-F1: {macro_f1:.4f}\n\n")
    f.write(f"Confusion:\n{cm}\n\n")
    f.write(f"Report:\n{rep}\n")

print("Modell gespeichert:", PROC / "model_nn.joblib")
print("Ergebnisse gespeichert:", PROC / "results_nn.txt")

# Konfusionsmatrix als PNG
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=range(1,6), yticklabels=range(1,6))
plt.xlabel("Vorhergesagte Klasse")
plt.ylabel("Wahre Klasse")
plt.title("Konfusionsmatrix – Neuronales Netz")
out_path = PROC / "confusion_nn.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()

print("Konfusionsmatrix gespeichert:", out_path)