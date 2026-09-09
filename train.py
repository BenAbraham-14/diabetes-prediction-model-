import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# 1. Load data
df = pd.read_csv("C:/Users/chris/Downloads/diabetes_data_upload.csv")
df = df.drop_duplicates()

# 2. Map binary text to numbers
mapping = {"Yes": 1, "No": 0, "Positive": 1, "Negative": 0, "Male": 1, "Female": 0}
df_numeric = df.replace(mapping)

features = [
    "Age",
    "Gender",
    "Polyuria",
    "Polydipsia",
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

# 3. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Train a Random Forest (robust, rule-based, no positive bias)
model = RandomForestClassifier(
    n_estimators=100, max_depth=6, min_samples_leaf=2, random_state=42
)
model.fit(X_train, y_train)

# 5. Evaluate
y_pred = model.predict(X_test)
print(f"Test Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 6. Test a clean "All No" synthetic patient right here in the terminal
test_healthy_male = pd.DataFrame(
    [
        {
            "Age": 45,
            "Gender": 1,
            "Polyuria": 0,
            "Polydipsia": 0,
            "sudden weight loss": 0,
            "weakness": 0,
            "Polyphagia": 0,
            "Genital thrush": 0,
            "visual blurring": 0,
            "Itching": 0,
            "Irritability": 0,
            "delayed healing": 0,
            "partial paresis": 0,
            "muscle stiffness": 0,
            "Alopecia": 0,
            "Obesity": 0,
        }
    ]
)[features]

healthy_prob = model.predict_proba(test_healthy_male)[0][1]
print(
    f"\nVerification -> Risk probability for all 'No' (Age 45, Male): {healthy_prob * 100:.2f}%"
)
# You should see ~3% to 8% here!

# 7. Save
joblib.dump({"model": model, "features": features}, "model.joblib")
print("Saved new model.joblib successfully!")