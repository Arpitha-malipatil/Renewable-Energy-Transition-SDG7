import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def run_linear_regression(csv_path: str):
    """
    Trains a Linear Regression model to predict
    renewable electricity share (%) using energy indicators.

    Parameters:
        csv_path (str): Path to CSV file

    Returns:
        model: Trained Linear Regression model
        r2 (float): R-squared score on test data
        predictions (array): Predicted values for test data
        y_test (array): Actual values from test data
    """

    # Load dataset
    data = pd.read_csv(csv_path)

    # Drop missing values
    data = data.dropna()

    # Define Features (X)
    X = data[[
        "Renewable_electricity_output_GWh_[4.1.2_REN.ELECTRICITY.OUTPUT]",
        "Total_electricity_output_GWh_[4.1.1_TOTAL.ELECTRICITY.OUTPUT]",
        "Time"
    ]]

    # Define Target (y)
    y = data[
        "Renewable_electricity_share_of_total_electricity_output_percent_[4.1_SHARE.RE.IN.ELECTRICITY]"
    ]

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train Model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict
    predictions = model.predict(X_test)

    # Evaluate
    r2 = r2_score(y_test, predictions)

    return model, r2, predictions, y_test


# Run directly
if __name__ == "__main__":
    model, score, preds, actuals = run_linear_regression("preprocessed_energy_data.csv")
    print("R2 Score:", round(score, 4))
    print("\nPredicted Values:\n", preds)
    print("\nActual Values:\n", actuals.values)