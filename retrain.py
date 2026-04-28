import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load data
heart_data = pd.read_csv('heart_disease_data.csv')

X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=2
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)

# Save fresh model
pickle.dump(model, open('trained_model.sav', 'wb'))
print("Model saved successfully!")