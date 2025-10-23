from pathlib import Path
from scipy.sparse import load_npz
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

BASE = Path(__file__).resolve().parent.parent
PROC = BASE / "data" / "processed"

X_train = load_npz(PROC / "X_train_tfidf.npz")
X_test  = load_npz(PROC / "X_test_tfidf.npz")
y_train = joblib.load(PROC / "y_train.joblib")
y_test  = joblib.load(PROC / "y_test.joblib")

clf = LogisticRegression(
    solver="liblinear",
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
cm  = confusion_matrix(y_test, y_pred)
rep = classification_report(y_test, y_pred, digits=3)

print("Accuracy:", round(acc, 4))
print("Confusion:\n", cm)
print("Report:\n", rep)

joblib.dump(clf, PROC / "model_logreg.joblib")
with open(PROC / "results_logreg.txt", "w") as f:
    f.write(f"Accuracy: {acc:.4f}\n\nConfusion:\n{cm}\n\nReport:\n{rep}\n")

print("Modell gespeichert:", PROC / "model_logreg.joblib")
print("Ergebnisse gespeichert:", PROC / "results_logreg.txt")

# Konfusionsmatrix als Diagramm speichern
import matplotlib.pyplot as plt
import seaborn as sns

labels = [1, 2, 3, 4, 5]

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels)

plt.xlabel("Vorhergesagte Klasse")
plt.ylabel("Wahre Klasse")
plt.title("Konfusionsmatrix Logistic Regression")

out_path = PROC / "confusion_logreg.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()

print("Konfusionsmatrix gespeichert:", out_path)