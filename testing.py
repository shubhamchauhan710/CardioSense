import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
data = pd.read_csv("heart_disease_data.csv")

# Split features and target
X = data.drop(columns="target", axis=1)
y = data["target"]

# Load trained model
model = pickle.load(open("trained_model.sav", "rb"))

# Predictions
y_pred = model.predict(X)

# Accuracy
accuracy = accuracy_score(y, y_pred)
print(f"\n✅ Accuracy: {accuracy * 100:.2f}%")

# Confusion Matrix
print("\n📊 Confusion Matrix:")
print(confusion_matrix(y, y_pred))

# Classification Report
print("\n📋 Classification Report:")
print(classification_report(y, y_pred))

# Example: show first 5 predictions vs actual
results = pd.DataFrame({
    "Actual": y.values,
    "Predicted": y_pred
})

print("\n🔍 Sample Predictions:")
print(results.head())