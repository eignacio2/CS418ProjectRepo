"""
popularity_rf.py

ML Analysis #1: Predict popularity class (Low / Medium / High)
using RandomForestClassifier on Spotify track audio features.

This script performs:
1. Load dataset.csv
2. Create popularity classes
3. Train/test split
4. Train RandomForestClassifier
5. Print accuracy & classification report
6. Plot confusion matrix (Matplotlib only)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)


def main():

    # 1. Load dataset
    df = pd.read_csv("data/raw/dataset.csv")

    # 2. Convert popularity (0–100) into classes: Low, Medium, High
    # using quantile-based binning for balanced classes
    df["popularity_class"] = pd.qcut(
        df["popularity"],
        q=3,
        labels=["Low", "Medium", "High"]
    )

    # 3. Select audio features
    feature_cols = [
        "danceability",
        "energy",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo"
    ]

    data = df[feature_cols + ["popularity_class"]].dropna()

    X = data[feature_cols]
    y = data["popularity_class"]

    # 4. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # 5. Train Random Forest Classifier
    clf = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )
    clf.fit(X_train, y_train)

    # 6. Predict & evaluate
    y_pred = clf.predict(X_test)

    print("=== Random Forest Popularity Classification ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    # 7. Confusion Matrix Plot (Matplotlib only)
    labels = ["Low", "Medium", "High"]
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    plt.figure(figsize=(6, 6))
    plt.imshow(cm)
    plt.title("Confusion Matrix – Random Forest (Popularity Class)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(ticks=[0, 1, 2], labels=labels)
    plt.yticks(ticks=[0, 1, 2], labels=labels)
    plt.colorbar()

    # Add numbers inside boxes
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
