import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import os

# 1. Load the final data
df = pd.read_csv('data/final/final_model_input.csv')

# 2. Define Features (Inputs) and Target (What we want to predict)
# We use last year's Solar, Wind, and Share to predict this year's Speed
features = ['Prev_Year_Solar_Cap', 'Prev_Year_Wind_Cap', 'Prev_Year_Renewable_Share']
X = df[features]
y = df['Transition_Speed']

# 3. Split data (80% for training, 20% for testing the model)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train the Model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Test the Model
y_pred = model.predict(X_test)

# 6. Print Results
print("--- MODEL RESULTS ---")
print(f"R2 Score (Accuracy): {r2_score(y_test, y_pred):.4f}")
print(f"Error (MSE): {mean_squared_error(y_test, y_pred):.4f}")

# 7. Create a Chart of "Importance"
plt.figure(figsize=(10,6))
plt.bar(features, model.coef_)
plt.title("Which factor drives Transition Speed the most?")
os.makedirs('results', exist_ok=True)
plt.savefig('results/feature_importance.png')
print("Chart saved in 'results/feature_importance.png'")