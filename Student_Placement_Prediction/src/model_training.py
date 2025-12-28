# import os
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score
# import joblib

# # ✅ Absolute path fix
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "student_placement_processed.csv")
# MODEL_DIR = os.path.join(BASE_DIR, "models")

# os.makedirs(MODEL_DIR, exist_ok=True)

# # Load data
# df = pd.read_csv(DATA_PATH)
# print("Processed data loaded successfully!")
import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ===============================
# Path handling (IMPORTANT)
# ===============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR, "data", "processed", "student_placement_processed.csv"
)
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

# ===============================
# Load processed data
# ===============================
df = pd.read_csv(DATA_PATH)
print("Processed data loaded successfully!")

# ===============================
# Features & Target
# ===============================
X = df.drop("status", axis=1)
y = df["status"]

# Encode target (Placed / Not Placed)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# ===============================
# Scale features
# ===============================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ===============================
# Train-Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ===============================
# Train Model
# ===============================
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ===============================
# Evaluation
# ===============================
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy:.2f}")

# ===============================
# Save model & preprocessors
# ===============================
joblib.dump(model, os.path.join(MODEL_DIR, "placement_model.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
joblib.dump(label_encoder, os.path.join(MODEL_DIR, "label_encoder.pkl"))

print("Model and preprocessing files saved successfully!")
