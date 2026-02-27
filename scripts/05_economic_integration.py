import pandas as pd
import numpy as np
import os

# 1. Load your processed data
df = pd.read_csv('data/final/final_model_input.csv')

# 2. Add Economic Variables (Solar Cost & Subsidies)
# We simulate a decreasing cost of solar and varying subsidies over time
df = df.sort_values(['Entity', 'Year'])

# Solar costs drop globally over time (simulated trend)
df['Solar_Cost_USD_Watt'] = 4.0 * (0.90 ** (df['Year'] - df['Year'].min())) 

# Government Subsidies (simulated based on previous renewable share)
df['Gov_Subsidy_M'] = (df['Prev_Year_Renewable_Share'] * 10) + np.random.normal(50, 20, len(df))

# 3. FEATURE ENGINEERING: Create the "Lagged Subsidy" (t-1)
# This represents the policy impact from the PREVIOUS year
df['Lagged_Subsidy'] = df.groupby('Entity')['Gov_Subsidy_M'].shift(1)

# 4. Save the Final Academic Dataset
df = df.dropna(subset=['Lagged_Subsidy'])
df.to_csv('data/final/academic_energy_data.csv', index=False)

print("SUCCESS: Economic features and Lagged Subsidies integrated!")