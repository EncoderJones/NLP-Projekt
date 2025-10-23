# Numerische Textklassifizierung von Produktrezensionen (NLP-Projekt)

Dieses Repository enthält den vollständigen Quellcode (`src/`) des Projekts *Numerische Textklassifizierung von Produktrezensionen*

## 📘 Projektbeschreibung
Ziel des Projekts ist die automatische Klassifikation von Amazon-Produktrezensionen anhand ihrer Texte in numerische Bewertungsstufen (1–5 Sterne) sowie eine vereinfachte Sentimentanalyse (negativ, neutral, positiv).  
Zur Umsetzung wurden Verfahren des überwachten maschinellen Lernens eingesetzt: **Naive Bayes**, **Logistic Regression** und ein **neuronales Netz**.

Die Datengrundlage bildete der **Amazon Reviews 2023 Datensatz** (Hou et al., 2023), bereitgestellt vom BLaIR-Team.

## 📂 Repository-Inhalt
- `src/` – enthält alle Python-Skripte zur Datenaufbereitung, Modellierung, Evaluation und Visualisierung.  
- `.gitignore` – definiert, welche Dateien und Ordner (z. B. große Daten oder Systemdateien) von der Versionskontrolle ausgeschlossen werden.

## 🚫 Nicht enthaltene Dateien
Die folgenden Daten wurden aufgrund ihrer Größe **nicht in das Repository aufgenommen**:
- Rohdaten (`data/raw/`)
- Verarbeitete Datensätze (`data/processed/`)
- Modell- und Vektorisierungsdateien (`.joblib`, `.npz`)
- Diagramme und Zwischenergebnisse (`.png`, `.csv`)

Diese Dateien wurden **lokal verarbeitet und zur Ergebnisgewinnung verwendet**, sind aber nicht notwendig, um den Quellcode oder das Vorgehen nachzuvollziehen.  
Alle Ergebnisse (z. B. Konfusionsmatrizen, Metriken, Abbildungen) sind im schriftlichen Projektbericht dokumentiert.

## 🧠 Anforderungen
- Python 3.10 oder höher  
- Notwendige Bibliotheken:
  ```bash
  pandas
  numpy
  scikit-learn
  matplotlib
  spacy
  joblib
