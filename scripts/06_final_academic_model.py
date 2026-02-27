import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import os

# 1. Load the new data
df = pd.read_csv('data/final/academic_energy_data.csv')

# 2. Define Academic Features (from your plan)
features = ['Solar_Cost_USD_Watt', 'Gov_Subsidy_M', 'Lagged_Subsidy', 'Prev_Year_Renewable_Share']
X = df[features]
y = df['Transition_Speed']

# 3. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Random Forest Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluation
y_pred = model.predict(X_test)
print(f"Academic Model R2 Score: {r2_score(y_test, y_pred):.4f}")
print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.4f}")

# 6. Feature Importance Visualization
importances = pd.Series(model.feature_importances_, index=features)
importances.sort_values().plot(kind='barh', color='darkgreen')
plt.title('Impact of Economic vs. Lagged Policy Features')
os.makedirs('results', exist_ok=True)
plt.savefig('results/academic_importance.png')
print("Final chart saved in results/academic_importance.png")