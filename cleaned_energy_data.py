import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


file_path = "cleaned_energy_data (1).csv.crdownload"
df = pd.read_csv("cleaned_energy_data (1).csv.crdownload")

print("Initial Shape:", df.shape)
print(df.head())


df.columns = (
    df.columns
    .str.strip()
    .str.replace(" ", "_")
    .str.replace("(", "", regex=False)
    .str.replace(")", "", regex=False)
    .str.replace("%", "percent")
)


if "Time" in df.columns:
    df["Time"] = pd.to_numeric(df["Time"], errors="coerce")

# Convert numeric columns properly
numeric_cols = df.select_dtypes(include=["object"]).columns
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="ignore")

print("\nAfter Formatting:")
print(df.dtypes)



df.fillna(method="ffill", inplace=True)

df.fillna(df.select_dtypes(include=np.number).mean(), inplace=True)

print("\nMissing Values After Handling:")
print(df.isnull().sum())

