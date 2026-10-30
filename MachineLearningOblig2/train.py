from pathlib import Path
import csv
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline

# Hent inn datasett
folder = Path(__file__).resolve().parent
data = pd.read_csv(
    folder / "data" / "amazon_cells_labelled.txt",
    sep="\t", names=["review", "sentiment"], quoting=csv.QUOTE_NONE
)

# Rydd opp i datasettet
data["review"] = data["review"].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
conflicts = data.groupby("review")["sentiment"].transform("nunique") > 1
data = data.loc[~conflicts].drop_duplicates("review")

# Distruber 20% til slutt-testen og 80% til trening og validering
#Stratify slik at vi har lik antall positive og negative
X_train, X_test, y_train, y_test = train_test_split(
    data["review"], data["sentiment"],
    test_size=0.2, stratify=data["sentiment"], random_state=42
)

#Presenter teksten i datasettet som TF-IDF til modellen vår som er av logistisk regresjon
#Sett opp pipeline for å sikre korrekt opptrening etter oversettelse til TF-IDF
pipeline = make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=1000))

#Kryssvalidering(Femfold), vi skal validere på femte
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Sammenligning for å oppnå beste innstillinger
search = GridSearchCV(
    pipeline,
    {
        "tfidfvectorizer__ngram_range": [(1, 1), (1, 2)],
        "logisticregression__C": [0.1, 1, 10],
    },
    cv=cv,
    scoring="accuracy"
)

search.fit(X_train, y_train)

# Vis valideringsresultatet etter hver innstilling
print("Antall etter rydding:", len(data))
print("Trening / test:", len(X_train), len(X_test))
for settings, score in zip(search.cv_results_["params"], search.cv_results_["mean_test_score"]):
    print(f"{settings}: validering = {score:.1%}")
print("Valgt innstilling:", search.best_params_)


predictions = search.predict(X_test)

# Baseline
baseline = [y_train.mode().iloc[0]] * len(y_test)
print(f"Baseline på testdata: {accuracy_score(y_test, baseline):.1%}")
print(f"Modell på testdata: {accuracy_score(y_test, predictions):.1%}")

# Vis verdiene for metrikkene våre
print(classification_report(
    y_test, predictions, labels=[0, 1],
    target_names=["Negativ", "Positiv"], zero_division=0
))

#Forvekslingsmatrisen
print("Forvekslingsmatrise: rader=fasit, kolonner=prediksjon; rekkefølge 0, 1")
print(confusion_matrix(y_test, predictions, labels=[0, 1]))

# Lagre
joblib.dump(search.best_estimator_, folder / "model.joblib")