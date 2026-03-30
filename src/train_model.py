import os
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

DATA_URL = (
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
    "pima-indians-diabetes.data.csv"
)

COLUMNS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
]

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
RANDOM_STATE = 42

def load_data(source=DATA_URL):

    df = pd.read_csv(source, names=COLUMNS)
    print(f"Loaded {len(df)} patient records.")
    return df

def fix_missing_values(df):
 
    columns_with_hidden_nulls = [
        "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"
    ]
    for col in columns_with_hidden_nulls:
        median_val = df[col].replace(0, np.nan).median()
        df[col] = df[col].replace(0, median_val)
        print(f"  Fixed '{col}' — replaced zeros with median ({median_val:.1f})")
    return df

def add_features(df):

    df = df.copy()
    df["BMI_Age"] = df["BMI"] * df["Age"]
    df["Glucose_Insulin_Ratio"] = df["Glucose"] / (df["Insulin"] + 1)
    return df

def train_and_evaluate(df):
    df = add_features(df)
    features = [col for col in df.columns if col != "Outcome"]
    X = df[features]
    y = df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_split=5,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train_scaled, y_train)

    y_pred  = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    print("\n── Results ──────────────────────────────────────")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC  : {roc_auc_score(y_test, y_proba):.4f}")

    cv = cross_val_score(model, scaler.transform(X), y, cv=5, scoring="roc_auc")
    print(f"CV AUC   : {cv.mean():.4f} ± {cv.std():.4f}  (5-fold)")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Diabetes", "Diabetes"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return model, scaler, features

def save_model(model, scaler, features):
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, "model.pkl"),    "wb") as f: pickle.dump(model,    f)
    with open(os.path.join(MODEL_DIR, "scaler.pkl"),   "wb") as f: pickle.dump(scaler,   f)
    with open(os.path.join(MODEL_DIR, "features.pkl"), "wb") as f: pickle.dump(features, f)
    print(f"\nModel saved to '{MODEL_DIR}/'")

if __name__ == "__main__":
    print("Starting training...\n")
    df = load_data()
    df = fix_missing_values(df)
    model, scaler, features = train_and_evaluate(df)
    save_model(model, scaler, features)
    print("\nDone. Now run: streamlit run app.py")
