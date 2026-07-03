"""
train_model.py
---------------
Reproduces the EXACT preprocessing + modeling logic from the original
Heart Disease Prediction notebook and serializes everything needed for
inference (model, scaler, column schema) into `model.pkl`.

ML LOGIC IS UNCHANGED from the notebook:
  1. Zero-value imputation for Cholesterol and RestingBP (replace 0 with
     the mean of non-zero values, rounded to 2 decimals).
  2. One-hot encoding via pd.get_dummies on the full dataframe.
  3. StandardScaler fit on ['Age','RestingBP','Cholesterol','MaxHR','Oldpeak'].
  4. 80/20 train_test_split with random_state=42.
  5. Five candidate models compared on accuracy_score:
     Logistic Regression, KNN, Naive Bayes, Decision Tree, SVM.
     Logistic Regression was the top performer in the notebook run
     (~0.87 accuracy) and is therefore the production model.

Usage:
    python train_model.py --data heart.csv
"""

import argparse
import pickle
import warnings

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

NUMERIC_COLS = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
CATEGORICAL_COLS = [
    "Sex",
    "ChestPainType",
    "RestingECG",
    "ExerciseAngina",
    "ST_Slope",
]
TARGET_COL = "HeartDisease"


def load_and_clean(path: str):
    """Load raw CSV and apply the notebook's zero-value imputation."""
    df = pd.read_csv(path)

    cholesterol_mean = df.loc[df["Cholesterol"] != 0, "Cholesterol"].mean()
    df["Cholesterol"] = df["Cholesterol"].replace(0, cholesterol_mean)
    df["Cholesterol"] = df["Cholesterol"].round(2)

    resting_mean = df.loc[df["RestingBP"] != 0, "RestingBP"].mean()
    df["RestingBP"] = df["RestingBP"].replace(0, resting_mean)
    df["RestingBP"] = df["RestingBP"].round(2)

    return df, cholesterol_mean, resting_mean


def encode_and_scale(df: pd.DataFrame):
    """One-hot encode + scale exactly like the notebook."""
    df_encode = pd.get_dummies(df)
    df_encode = df_encode.astype(int)

    scaler = StandardScaler()
    df_encode[NUMERIC_COLS] = scaler.fit_transform(df[NUMERIC_COLS])

    X = df_encode.drop(TARGET_COL, axis=1)
    y = df_encode[TARGET_COL]
    return X, y, scaler


def train_and_select_best(X, y):
    """Train the 5 candidate models and pick the best by accuracy."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "KNN": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(),
        "SVM": SVC(probability=True),
    }

    results = []
    fitted = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results.append((name, acc))
        fitted[name] = model

    results.sort(key=lambda r: r[1], reverse=True)
    best_name, best_acc = results[0]
    best_model = fitted[best_name]

    print("Model comparison (test accuracy):")
    for name, acc in results:
        marker = "  <-- selected" if name == best_name else ""
        print(f"  {name:22s} {acc:.4f}{marker}")

    return best_model, best_name, best_acc, results, (X_test, y_test)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", default="heart.csv", help="Path to the heart.csv dataset"
    )
    parser.add_argument(
        "--out", default="model.pkl", help="Output path for the pickled bundle"
    )
    args = parser.parse_args()

    df, cholesterol_mean, resting_mean = load_and_clean(args.data)
    X, y, scaler = encode_and_scale(df)

    best_model, best_name, best_acc, results, (X_test, y_test) = train_and_select_best(
        X, y
    )

    bundle = {
        "model": best_model,
        "model_name": best_name,
        "scaler": scaler,
        "feature_columns": list(X.columns),
        "numeric_cols": NUMERIC_COLS,
        "categorical_cols": CATEGORICAL_COLS,
        "cholesterol_mean": round(float(cholesterol_mean), 2),
        "resting_bp_mean": round(float(resting_mean), 2),
        "all_model_results": results,
        "test_accuracy": best_acc,
        "n_train_samples": len(X) - len(X_test),
        "n_test_samples": len(X_test),
    }

    with open(args.out, "wb") as f:
        pickle.dump(bundle, f)

    print(f"\nSaved model bundle -> {args.out}")
    print(f"Selected model: {best_name} (accuracy={best_acc:.4f})")


if __name__ == "__main__":
    main()
