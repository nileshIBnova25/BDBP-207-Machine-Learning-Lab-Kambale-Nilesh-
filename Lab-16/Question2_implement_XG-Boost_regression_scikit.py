#-------------------Imports----------------#
import xgboost as xgb
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#-----------------------------Load-data---------------------------#
data = fetch_california_housing()
X, y = data.data, data.target

#------------Train-Test-Split-------------------------------------#
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def main():
    # Initialize Regressor
    reg_model = xgb.XGBRegressor(
        n_estimators=100, # n_estimators: Number of boosting rounds
        learning_rate=0.1, # learning_rate: Step size shrinkage to prevent overfitting,
        max_depth=5,
        random_state=42
    )

    # Fit and Predict
    reg_model.fit(X_train, y_train)
    y_pred = reg_model.predict(X_test)

    print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
    print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")

if __name__ == '__main__':
    main()