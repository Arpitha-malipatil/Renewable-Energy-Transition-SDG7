import pandas as pd
import numpy as np
import os

# 1. Define the original 17 files
all_files = [
    "01 renewable-share-energy.csv", "02 modern-renewable-energy-consumption.csv",
    "03 modern-renewable-prod.csv", "04 share-electricity-renewables.csv",
    "05 hydropower-consumption.csv", "06 hydro-share-energy.csv",
    "07 share-electricity-hydro.csv", "08 wind-generation.csv",
    "09 cumulative-installed-wind-energy-capacity-gigawatts.csv", "10 wind-share-energy.csv",
    "11 share-electricity-wind.csv", "12 solar-energy-consumption.csv",
    "13 installed-solar-PV-capacity.csv", "14 solar-share-energy.csv",
    "15 share-electricity-solar.csv", "16 biofuel-production.csv",
    "17 installed-geothermal-capacity.csv"
]

raw_path = 'data/raw/' 
os.makedirs('data/processed', exist_ok=True)

print("Step 1: Merging the 17 core energy files...")
master_df = pd.read_csv(os.path.join(raw_path, all_files[0]))

for file in all_files[1:]:
    file_path = os.path.join(raw_path, file)
    if os.path.exists(file_path):
        df_temp = pd.read_csv(file_path)
        master_df = pd.merge(master_df, df_temp, on=['Entity', 'Code', 'Year'], how='outer')

# --- INTEGRATING THE 4 NEW FILES ---
print("Step 2: Adding supplementary economic and solar data...")

# 18 & 19. Process data.csv and metadata.csv (Same format)
for extra_file in ['data.csv', 'metadata.csv']:
    if os.path.exists(extra_file):
        # Convert ".." to NaN so we can do math
        df_extra = pd.read_csv(extra_file, na_values="..")
        df_extra = df_extra.rename(columns={
            'Country Name': 'Entity', 
            'Country Code': 'Code', 
            'Time': 'Year'
        })
        # Clean Year column
        df_extra = df_extra.dropna(subset=['Year'])
        df_extra['Year'] = df_extra['Year'].astype(int)
        
        # Merge into master
        master_df = pd.merge(master_df, df_extra, on=['Entity', 'Code', 'Year'], how='outer')

# 20. Process renewable_energy.csv (Needs pivoting)
if os.path.exists('renewable_energy.csv'):
    df_re = pd.read_csv('renewable_energy.csv')
    df_re = df_re.rename(columns={'LOCATION': 'Code', 'TIME': 'Year'})
    df_re_pivot = df_re.pivot_table(index=['Code', 'Year'], columns='MEASURE', values='Value').reset_index()
    master_df = pd.merge(master_df, df_re_pivot, on=['Code', 'Year'], how='outer')

# 21. Process solar-pv-prices.csv (Solar panel cost data)
if os.path.exists('solar-pv-prices.csv'):
    df_prices = pd.read_csv('solar-pv-prices.csv')
    master_df = pd.merge(master_df, df_prices, on=['Entity', 'Code', 'Year'], how='outer')

# --- FINAL CLEANUP ---
# Fix any column name overlaps (like Entity_x / Entity_y)
for col in master_df.columns:
    if col.endswith('_x'):
        base = col[:-2]
        other = base + '_y'
        if other in master_df.columns:
            master_df[base] = master_df[col].fillna(master_df[other])
            master_df.drop([col, other], axis=1, inplace=True)

# Sort and Save
master_df = master_df.sort_values(by=['Entity', 'Year']).reset_index(drop=True)
master_df.to_csv('data/processed/master_renewable_dataset.csv', index=False)

print(f"Success! Merged 21 files. Final shape: {master_df.shape}")