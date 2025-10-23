from pathlib import Path
import joblib, numpy as np
from scipy.sparse import load_npz
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

BASE = Path(__file__).resolve().parent.parent
PROC = BASE / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)


X_train = load_npz(PROC / "X_train_tfidf.npz")
X_test  = load_npz(PROC / "X_test_tfidf.npz")
y_train = joblib.load(PROC / "y_train_bin.joblib")
y_test  = joblib.load(PROC / "y_test_bin.joblib")

def train_nb():
    return MultinomialNB().fit(X_train, y_train)

def train_logreg():
    return LogisticRegression(solver="liblinear",
                              class_weight="balanced",
                              max_iter=1000,
                              random_state=42).fit(X_train, y_train)

def train_nn():
    return MLPClassifier(hidden_layer_sizes=(128,),
                         activation="relu",
                         solver="adam",
                         alpha=1e-3,
                         learning_rate_init=1e-3,
                         max_iter=200,
                         early_stopping=True,
                         validation_fraction=0.1,
                         n_iter_no_change=10,
                         random_state=42,
                         verbose=False).fit(X_train, y_train)

def evaluate_and_save(name, clf):
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    cm = confusion_matrix(y_test, y_pred)
    rep = classification_report(y_test, y_pred, digits=3)

    print(f"\n{name}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")
    print("Confusion:\n", cm)
    print("Report:\n", rep)

    # Modell
    joblib.dump(clf, PROC / f"model_{name}.joblib")
    # Text-Report
    with open(PROC / f"results_{name}.txt", "w") as f:
        f.write(f"{name}\nAccuracy: {acc:.4f}\nMacro-F1: {macro_f1:.4f}\n\n")
        f.write(f"Confusion:\n{cm}\n\nReport:\n{rep}\n")

    # Plot Konfusionsmatrix
    labels = sorted(np.unique(y_test))
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Vorhergesagte Klasse"); plt.ylabel("Wahre Klasse")
    plt.title(f"Konfusionsmatrix – {name}")
    plt.savefig(PROC / f"confusion_{name}.png", dpi=300, bbox_inches="tight")
    plt.close()

def main():
    models = {
        "nb_bin": train_nb,
        "logreg_bin": train_logreg,
        "nn_bin": train_nn
    }
    for name, trainer in models.items():
        clf = trainer()
        evaluate_and_save(name, clf)

if __name__ == "__main__":
    main()