import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# Load dataset
df = pd.read_csv("diabetes.csv")


# Features and target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)


# Build model
model = SVC(kernel="linear")


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
with open("diabetes.pkl", "wb") as file:
    pickle.dump(model, file)


print("Diabetes model saved successfully.")