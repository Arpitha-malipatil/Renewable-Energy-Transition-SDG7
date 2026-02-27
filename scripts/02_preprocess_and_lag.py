import pandas as pd
import numpy as np
import os

# 1. Load the data created by your first script
input_path = 'data/processed/master_renewable_dataset.csv'
if not os.path.exists(input_path):
    print(f"Error: {input_path} not found. Run 01_merge.py first!")
else:
    df = pd.read_csv(input_path)

    # 2. Filter for real countries only (3-letter codes)
    df = df[df['Code'].notna()].copy()
    df = df[df['Code'].str.len() == 3]

    # 3. Fill missing values with 0
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    # 4. Create "Transition_Speed" and Lags
    df = df.sort_values(['Entity', 'Year'])
    df['Transition_Speed'] = df.groupby('Entity')['Renewables (% electricity)'].diff()

    # Look-back features (Lags)
    df['Prev_Year_Solar_Cap'] = df.groupby('Entity')['Solar Capacity'].shift(1)
    df['Prev_Year_Wind_Cap'] = df.groupby('Entity')['Wind Capacity'].shift(1)
    df['Prev_Year_Renewable_Share'] = df.groupby('Entity')['Renewables (% electricity)'].shift(1)

    # 5. Remove the "empty" rows caused by the shift
    df = df.dropna(subset=['Transition_Speed', 'Prev_Year_Solar_Cap'])

    # 6. SAVE - This creates the file you are looking for!
    os.makedirs('data/final', exist_ok=True)
    df.to_csv('data/final/final_model_input.csv', index=False)
    print("SUCCESS: 'data/final/final_model_input.csv' has been created!")