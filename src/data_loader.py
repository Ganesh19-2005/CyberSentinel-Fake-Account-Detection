import pandas as pd

# Load dataset
data = pd.read_csv("data/social_accounts.csv")

print("\n===== FIRST FIVE ACCOUNTS =====")
print(data.head())

print("\n===== DATASET SHAPE =====")
print(data.shape)

print("\n===== COLUMN INFORMATION =====")
print(data.info())

print("\n===== FAKE AND GENUINE COUNTS =====")
print(data["label"].value_counts())

print("\n===== MISSING VALUES =====")
print(data.isnull().sum())

print("\nDataset loaded successfully!")
