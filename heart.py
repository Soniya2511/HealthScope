import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Load dataset
df = pd.read_csv("heart.csv")


# Features and target
X = df.drop("target", axis=1)
y = df["target"]


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)


# Build model
model = LogisticRegression(max_iter=1000)


# Train model
model.fit(X_train, y_train)


# Predictions
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)


# Accuracy
train_accuracy = accuracy_score(y_train, train_predictions)
test_accuracy = accuracy_score(y_test, test_predictions)


print(f"Training Accuracy : {train_accuracy * 100:.2f}%")
print(f"Testing Accuracy  : {test_accuracy * 100:.2f}%")


# Save trained model
with open("heart.pkl", "wb") as file:
    pickle.dump(model, file)


print("Heart disease model saved successfully.")