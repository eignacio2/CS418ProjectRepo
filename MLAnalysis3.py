import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score

# Load cleaned dataset
df = pd.read_csv("spotify_2023_clean.csv")

# Create the hit label based on the top 25% popularity
hit_threshold = df["track_popularity"].quantile(0.75)
df["is_hit"] = (df["track_popularity"] >= hit_threshold).astype(int)

# Select features for the model:
# duration and explicit flag
# release month
df["added_at"] = pd.to_datetime(df["added_at"], errors="coerce")
df["month"] = df["added_at"].dt.month

month_dummies = pd.get_dummies(df["month"], prefix="month")

X = pd.concat([
    df[["duration_ms", "explicit"]],
    month_dummies
], axis=1)

y = df["is_hit"]

# Train-test split, keeping the hit ratio consistent
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features because kNN relies on distance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train and evaluate kNN models with different k values
results = {}
for k in [3, 5, 11]:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    recall_hit = recall_score(y_test, y_pred, pos_label=1)
    cm = confusion_matrix(y_test, y_pred)

    results[k] = {
        "accuracy": acc,
        "hit_recall": recall_hit,
        "confusion_matrix": cm
    }

# Baseline accuracy
majority_class = y_train.mode()[0]
baseline_accuracy = (y_test == majority_class).mean()

print("BASELINE ACCURACY:", baseline_accuracy)
print("\n--- kNN RESULTS ---")

for k, r in results.items():
    print(f"\nk = {k}")
    print("Accuracy:", r["accuracy"])
    print("Hit Recall:", r["hit_recall"])
    print("Confusion Matrix:\n", r["confusion_matrix"])