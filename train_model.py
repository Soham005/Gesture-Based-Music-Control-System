import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

print("Loading dataset...")

df = pd.read_csv("gestures.csv", header=None)

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

print("Training model...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
)

model = RandomForestClassifier(n_estimators=200)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

print(f"Model Accuracy: {accuracy*100:.2f}%")

joblib.dump(model, "gesture_model.pkl")

print("Model saved successfully ✅")