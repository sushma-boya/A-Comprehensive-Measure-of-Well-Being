# ============================================
# Human Development Index (HDI) Predictor
# Model Training Script
# ============================================

import os
import pickle
import warnings

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ============================================
# Load Dataset
# ============================================

df = pd.read_csv("dataset/hdi_dataset.csv", encoding="latin1")

# Remove unwanted spaces from column names
df.columns = df.columns.str.strip()

print("\nDataset Loaded Successfully!")
print(df.head())

# ============================================
# Remove Unnecessary Columns
# ============================================

df.drop(
    columns=[
        "Unnamed: 10",
        "Unnamed: 11",
        "HUMAN DEVELOPMENT"
    ],
    inplace=True,
    errors="ignore"
)

# ============================================
# Replace '..' with Missing Values
# ============================================

df.replace("..", pd.NA, inplace=True)

# ============================================
# Convert Numeric Columns
# ============================================

numeric_columns = [
    "Human Development Index (HDI)",
    "Life expectancy at birth",
    "Expected years of schooling",
    "Mean years of schooling",
    "Gross national income (GNI) per capita"
]

# Remove commas from GNI values
df["Gross national income (GNI) per capita"] = (
    df["Gross national income (GNI) per capita"]
    .astype(str)
    .str.replace(",", "", regex=False)
)

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

# ============================================
# Fill Missing Values
# ============================================

df[numeric_columns] = df[numeric_columns].fillna(
    df[numeric_columns].mean()
)

print("\nMissing Values:")
print(df[numeric_columns].isnull().sum())

# ============================================
# Select Features and Target
# ============================================

X = df[
    [
        "Life expectancy at birth",
        "Expected years of schooling",
        "Mean years of schooling",
        "Gross national income (GNI) per capita"
    ]
]

y = df["Human Development Index (HDI)"]

print(df["Human Development Index (HDI)"].describe())

# ============================================
# Train-Test Split
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Samples :", X_train.shape[0])
print("Testing Samples  :", X_test.shape[0])

# ============================================
# Train Linear Regression Model
# ============================================

model = LinearRegression()

model.fit(X_train, y_train)

print("\nModel Training Completed!")

print("Training R²:", model.score(X_train, y_train))
print("Testing R² :", model.score(X_test, y_test))
# ============================================
# Prediction
# ============================================

y_pred = model.predict(X_test)

# ============================================
# Model Evaluation
# ============================================

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5

print("\n========== MODEL PERFORMANCE ==========")
print(f"R² Score               : {r2:.4f}")
print(f"Mean Absolute Error    : {mae:.4f}")
print(f"Mean Squared Error     : {mse:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")

# ============================================
# Save Model
# ============================================

os.makedirs("models", exist_ok=True)

with open("models/hdi_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("\nModel saved successfully!")
print("Location : models/hdi_model.pkl")