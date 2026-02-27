import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import os

# 1. Load the data
df = pd.read_csv('data/final/final_model_input.csv')

# 2. Define Features and Target
# Automatically grab all numeric columns except the one we want to predict
features = [col for col in df.columns if df[col].dtype in ['float64', 'int64'] 
            and col not in ['Year', 'Transition_Speed', 'Renewables (% electricity)']]
X = df[features]
y = df['Transition_Speed']

# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train the Random Forest Model
# We use 100 trees to get a more stable prediction
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Predictions and Scoring
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("--- ADVANCED MODEL (RANDOM FOREST) RESULTS ---")
print(f"New R2 Score: {r2:.4f}")
print(f"Average Error (MAE): {mae:.4f} percentage points")

# 6. Better Feature Importance Chart
importances = pd.Series(model.feature_importances_, index=features)
importances.sort_values().plot(kind='barh', color='teal')
plt.title('Which Factors Actually Matter?')
plt.xlabel('Importance Score')

os.makedirs('results', exist_ok=True)
plt.savefig('results/advanced_importance.png')
print("\nSuccess! New chart saved in 'results/advanced_importance.png'")