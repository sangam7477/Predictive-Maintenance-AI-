import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ==========================================
# 1. Load Dataset
# ==========================================

df = pd.read_csv("dataset/ai4i2020.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# ==========================================
# 2. Select Features and Target
# ==========================================

features = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

X = df[features]
y = df["Machine failure"]

print("\nFeatures:")
print(features)

print("\nTarget:")
print("Machine failure")


# ==========================================
# 3. Check Target Distribution
# ==========================================

print("\nMachine Failure Distribution:")
print(y.value_counts())


# ==========================================
# 4. Split Dataset
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# 5. Create Random Forest Model
# ==========================================

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)


# ==========================================
# 6. Train Model
# ==========================================

print("\nTraining model...")

model.fit(X_train, y_train)

print("Model training completed!")


# ==========================================
# 7. Make Predictions
# ==========================================

y_pred = model.predict(X_test)


# ==========================================
# 8. Evaluate Model
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print("\n================================")
print("MODEL PERFORMANCE")
print("================================")

print("\nAccuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# ==========================================
# 9. Save Model
# ==========================================

joblib.dump(
    model,
    "models/predictive_maintenance_model.pkl"
)

print("\nModel saved successfully!")
print("Location: models/predictive_maintenance_model.pkl")


# ==========================================
# 10. Feature Importance
# ==========================================

importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n================================")
print("FEATURE IMPORTANCE")
print("================================")

print(importance)
# ==========================================
# 11. Save Visualizations
# ==========================================

# Create static folder if it does not exist
import os

os.makedirs("static", exist_ok=True)


# ------------------------------------------
# Confusion Matrix Graph
# ------------------------------------------

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Failure", "Failure"],
    yticklabels=["No Failure", "Failure"]
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()

plt.savefig("static/confusion_matrix.png")

plt.close()


# ------------------------------------------
# Feature Importance Graph
# ------------------------------------------

plt.figure(figsize=(8, 5))

sns.barplot(
    data=importance,
    x="Importance",
    y="Feature"
)

plt.title("Feature Importance")

plt.tight_layout()

plt.savefig("static/feature_importance.png")

plt.close()


# ------------------------------------------
# Machine Failure Distribution
# ------------------------------------------

plt.figure(figsize=(6, 5))

sns.countplot(
    data=df,
    x="Machine failure"
)

plt.title("Machine Failure Distribution")
plt.xlabel("Machine Failure")
plt.ylabel("Count")

plt.tight_layout()

plt.savefig("static/failure_distribution.png")

plt.close()

print("\nGraphs saved successfully!")

print("Location: static/")