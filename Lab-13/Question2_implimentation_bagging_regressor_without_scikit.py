#====================Import======================#
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
#================================================#

class ScratchBaggingRegressor:
    def __init__(self, n_estimators=100, max_depth=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.models = []

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        n_samples = X.shape[0]
        self.models = []

        for i in range(self.n_estimators):
            # 1. Create Bootstrap Sample (Sampling with replacement
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_bag = X[indices]
            y_bag = y[indices]

            # 2. Create a Brand New model instance inside the loop every time
            model = DecisionTreeRegressor(max_depth=self.max_depth)

            # 3. Train the model on this specific bag or data set
            model.fit(X_bag, y_bag)

            # 4. Store the trained model  list
            self.models.append(model)

    def predict(self, X):
        # 5. Collect predictions from all independent model
        all_preds = np.array([m.predict(X) for m in self.models])
        print(all_preds)

        # 6. Aggregate by taking the mean (averaging the perspectives)
        return np.mean(all_preds, axis=0)


# --- Execution ---

def main():
    #------------------------------Loading-Data---------------------------------#
    data = load_diabetes()
    X, y = data.data, data.target

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train our scratch model
    bagger = ScratchBaggingRegressor(n_estimators=1000, max_depth=5)
    bagger.fit(X_train, y_train)

    # Make predictions it will return aggrigate value for all modes predict
    y_pred = bagger.predict(X_test)

    # Calculate performance
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Results for Diabetes Dataset:")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R2 Score: {r2:.4f}")

if __name__ == "__main__":
    main()