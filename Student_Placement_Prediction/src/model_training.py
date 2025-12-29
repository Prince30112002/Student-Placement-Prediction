import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ==============================
# Load processed dataset
# ==============================
DATA_PATH = "Student_Placement_Prediction/data/processed/student_placement_processed.csv"
df = pd.read_csv(DATA_PATH)

print("Processed data loaded successfully!")
print(df.head())

# ==============================
# One-Hot Encoding (BEST FIX)
# ==============================
df_encoded = pd.get_dummies(df, drop_first=True)

# ==============================
# Features & Target
# ==============================
X = df_encoded.drop("Placed", axis=1)
y = df_encoded["Placed"]

# ==============================
# Train-test split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# Train model
# ==============================
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ==============================
# Evaluation
# ==============================
y_pred = model.predict(X_test)

print("\n✅ Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ==============================
# Save model
# ==============================
joblib.dump(model, "Student_Placement_Prediction/models/placement_model.pkl")

print("\n🎉 Model trained & saved successfully!")
