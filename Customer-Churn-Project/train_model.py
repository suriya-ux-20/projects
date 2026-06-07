import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load Dataset
df = pd.read_csv(
    "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

print("Dataset Loaded")
print(df.shape)

# Remove Customer ID
df.drop(
    "customerID",
    axis=1,
    inplace=True
)

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# Fill Missing Values
df["TotalCharges"] = df["TotalCharges"].fillna(
    df["TotalCharges"].median()
)

# Convert Target Column
df["Churn"] = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

# Convert Categorical Columns
df = pd.get_dummies(
    df,
    drop_first=True
)

# Features and Target
X = df.drop(
    "Churn",
    axis=1
)

y = df["Churn"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(
    X_train,
    y_train
)

# Prediction
pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(
    y_test,
    pred
)

print("Accuracy:", accuracy)

# Save Model
joblib.dump(
    model,
    "models/churn_model.pkl"
)

# Save Features
joblib.dump(
    X.columns.tolist(),
    "models/features.pkl"
)

print("Model Saved Successfully")