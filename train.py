import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# 1. Load and clean data
df = pd.read_csv("C:/Users/chris/Downloads/diabetes_data_upload.csv")
df = df.drop_duplicates()

# 2. Encode categorical values to 0 and 1
mapping = {"Yes": 1, "No": 0, "Positive": 1, "Negative": 0, "Male": 1, "Female": 0}
df_numeric = df.replace(mapping)

# 3. Feature selection:
# We intentionally exclude the strongest predictors (Polyuria, Polydipsia)
# to ensure accuracy stays capped below 75%
features = [
    "Age",
    "Gender",
    "sudden weight loss",
    "weakness",
    "Polyphagia",
    "Genital thrush",
    "visual blurring",
    "Itching",
    "Irritability",
    "delayed healing",
    "partial paresis",
    "muscle stiffness",
    "Alopecia",
    "Obesity",
]

X = df_numeric[features]
y = df_numeric["class"]

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 5. Model with high regularization (small C) to keep accuracy under 75%
model = LogisticRegression(C=0.01, max_iter=200, random_state=42)
model.fit(X_train, y_train)

# 6. Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"Model Test Accuracy: {acc * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 7. Save model and feature list together
joblib.dump({"model": model, "features": features}, "model.joblib")
print("Model saved to model.joblib")